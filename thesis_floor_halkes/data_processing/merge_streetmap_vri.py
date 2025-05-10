# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point
# from sklearn.neighbors import BallTree
# import numpy as np

# # Define threshold in meters
# threshold_meters = 10  # adjust as needed
# earth_radius = 6371000  # meters
# df1 = pd.read_csv("data/processed/intersection_metadata.csv")
# df2 = pd.read_csv("data/processed/coords_nodes_helmond.csv")
# df3 = pd.read_csv("data/processed/intersection_measurements_31_01_24.csv")
# df3 = df3[~df3["tlc_name"].str.contains("TEL")]
# df1 = df1.merge(df3, on="tlc_name", how="right")
# df1 = df1.loc[:, ["lat", "lon", "tlc_name", "wait_time_all_cycles_average"]]
# df1 = df1.loc[~df1["tlc_name"].str.contains("TEL"), :]

# # # Create GeoDataFrames
# df1['geometry'] = df1.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
# df2['geometry'] = df2.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
# gdf1 = gpd.GeoDataFrame(df1, geometry='geometry')
# gdf2 = gpd.GeoDataFrame(df2, geometry='geometry')

# # Convert to radians
# coords1 = np.radians(gdf1[['lat', 'lon']].values)
# coords2 = np.radians(gdf2[['lat', 'lon']].values)

# # Build BallTree
# tree = BallTree(coords2, metric='haversine')
# distances, indices = tree.query(coords1, k=1)
# distances_m = distances * earth_radius

# # Filter by distance threshold
# gdf1['match_index'] = indices.flatten()
# gdf1['distance'] = distances_m.flatten()
# gdf1_filtered = gdf1[gdf1['distance'] <= threshold_meters]
# print(gdf1)
# print(gdf2)
# print(gdf1_filtered.shape)

# # Join matched data
# matched = gdf1_filtered.merge(df2, left_on='match_index', right_index=True, suffixes=('_df1', '_df2'))
# print(matched.shape)


import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import BallTree

# 1) load and pre-filter
df_meta = pd.read_csv("data/processed/intersection_metadata.csv")
df_nodes = pd.read_parquet("data/processed/coords_nodes_helmond.parquet")
df_meas = pd.read_csv("data/processed/intersection_measurements_31_01_24.csv")
print(df_meta['tlc_name'].nunique())

# drop any tlc_name containing "TEL"
df_meas = df_meas[~df_meas["tlc_name"].str.contains("TEL", na=False)]

# merge metadata + measurements → lights
df_lights = (
    df_meta
    .merge(df_meas[["tlc_name","wait_time_all_cycles_average"]],
           on="tlc_name", how="inner")
    .rename(columns={"wait_time_all_cycles_average":"wait_time"})
    [["tlc_name","lat","lon","wait_time"]]
)

# 2) build GeoDataFrames in WGS84
gdf_lights = gpd.GeoDataFrame(
    df_lights,
    geometry=[Point(xy) for xy in zip(df_lights.lon, df_lights.lat)],
    crs="EPSG:4326"
)
gdf_nodes = gpd.GeoDataFrame(
    df_nodes,
    geometry=[Point(xy) for xy in zip(df_nodes.lon, df_nodes.lat)],
    crs="EPSG:4326"
)

# 3) prepare for haversine BallTree (radians)
coords_lights = np.radians(gdf_lights[["lat","lon"]].values)
coords_nodes  = np.radians(gdf_nodes[["lat","lon"]].values)

tree = BallTree(coords_lights, metric="haversine")

# 4) query: for each node, get nearest light
dist_rad, idx = tree.query(coords_nodes, k=1)
dist_m = dist_rad.flatten() * 6_371_000  # earth radius in meters

# attach to gdf_nodes
gdf_nodes["nearest_light_idx"] = idx.flatten()
gdf_nodes["distance_m"]        = dist_m

# pull in the tlc_name & wait_time columns
gdf_nodes["tlc_name"] = gdf_lights.iloc[gdf_nodes["nearest_light_idx"]].tlc_name.values
gdf_nodes["wait_time"] = gdf_lights.iloc[gdf_nodes["nearest_light_idx"]].wait_time.values

# 5) apply your 10 m threshold
threshold = 25  # meters
mask = gdf_nodes["distance_m"] <= threshold
gdf_nodes.loc[~mask, ["tlc_name","wait_time"]] = 0

# 6) final flags + select
gdf_nodes["has_light"] = mask.astype(int)

# --- A) extract the static node→light lookup (no time yet) ---
df_nodes_map = (
    gdf_nodes
    .loc[:, ["node_id","tlc_name","lat","lon","has_light", 'distance_m']]
    .copy()
)

# --- B) parse & prepare your 15-min measurements table ---
df_meas["timestamp"] = pd.to_datetime(df_meas["timestamp"])
# assume all measurements happen on the same date; build a full-day 15-min grid:
day = df_meas["timestamp"].dt.normalize().min()
times = pd.date_range(start=day,
                      end=day + pd.Timedelta(hours=23, minutes=45),
                      freq="15T")
df_times = pd.DataFrame({"timestamp": times})

# --- C) cross-join nodes × times, then bring in wait_time by tlc_name+timestamp ---
# 1) add a dummy key to both for cross-join
df_nodes_map["key"] = 1
df_times   ["key"] = 1

# 2) make the full‐cartesian product
df_cross = df_nodes_map.merge(df_times, on="key").drop("key", axis=1)

# 3) left-join your real measurements
df_final = (
    df_cross
    .merge(
       df_meas[["tlc_name","timestamp","wait_time_all_cycles_average"]],
       on=["tlc_name","timestamp"],
       how="left"
    )
    .rename(columns={"wait_time_all_cycles_average":"wait_time"})
    .assign(wait_time=lambda d: d["wait_time"].fillna(0))
    .sort_values(["node_id","timestamp"])
    .reset_index(drop=True)
)
df_final["tlc_name"] = df_final["tlc_name"].astype("string")

# inspect
print(df_final)
print(df_final.loc[df_final['node_id']==1])
print(df_final.loc[df_final['node_id']==3919])
print("nodes with light:", mask.sum(), "/", len(gdf_nodes))

df_final.to_parquet("data/processed/node_features.parquet", index=False)