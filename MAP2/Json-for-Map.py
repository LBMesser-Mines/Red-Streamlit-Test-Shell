import json
import pandas as pd 

#-----------------Parameters---------------------#
MActivityFileName = input("Please enter the MActivity.csv name: ")


#-----------------Loading Lat/Long---------------------#
with open("lstx-reasonable.json") as f:
	data = json.load(f)

df2 = pd.DataFrame(data["locations"])[["id", "latitude", "longitude"]]

import os

# -----------------Loading Routes---------------------#
df3 = pd.read_csv(MActivityFileName)
df3.columns = df3.columns.str.strip()

# per-row weight
df3["WeightT"] = df3[["Mvalue", "per-item weight"]].prod(axis=1)

# -----------------Prep Lat/Long Lookup (do once)---------------------#
df2["id"] = df2["id"].astype(str).str.strip()

origin_lookup = df2.rename(columns={
    "id": "l",
    "latitude": "originLat",
    "longitude": "originLong"
})

dest_lookup = df2.rename(columns={
    "id": "lprime",
    "latitude": "destLat",
    "longitude": "destLong"
})

# -----------------Loading Symbols---------------------#
df_symbols = pd.read_csv("symbol-injector-output.csv")
df_symbols.columns = df_symbols.columns.str.strip()
df_symbols["location"] = df_symbols["location"].astype(str).str.strip()

# Create symbol lookups for origin and destination
origin_symbol_lookup = df_symbols.rename(columns={
    "location": "l",
    "Symbol": "sym1"
})

dest_symbol_lookup = df_symbols.rename(columns={
    "location": "lprime",
    "Symbol": "sym2"
})

# Optional: output folder
out_dir = "routes_by_time"
os.makedirs(out_dir, exist_ok=True)

# -----------------Loop over all time periods---------------------#
for t_val, df_t in df3.groupby("t"):
    # aggregate within this time period
    df_agg = (
        df_t.groupby(["l", "lprime", "t"], as_index=False)
            .agg(totalWeight=("WeightT", "sum"))
    )

    # normalize route ids (do per t so it’s guaranteed consistent)
    df_agg["l"] = df_agg["l"].astype(str).str.strip()
    df_agg["lprime"] = df_agg["lprime"].astype(str).str.strip()

    # merges
    df_out = df_agg.merge(origin_lookup, on="l", how="left")
    df_out = df_out.merge(dest_lookup, on="lprime", how="left")
    df_out = df_out.merge(origin_symbol_lookup, on="l", how="left")
    df_out = df_out.merge(dest_symbol_lookup, on="lprime", how="left")

    # reorder
    df_out = df_out[[
        "l", "originLat", "originLong",
        "lprime", "destLat", "destLong",
        "t", "totalWeight",
        "sym1", "sym2"
    ]]

    # write one csv per time period (zero-pad so sorting is nice)
    df_out.to_csv(f"{out_dir}/routesTime{int(t_val):03d}.csv", index=False)

