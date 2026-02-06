import json
import pandas as pd 
import numpy as np

with open("lstx-reasonable.json") as f:
	data = json.load(f)

lPairs = data["location_pair_data"]

rows = [
	
	{
        "Origin": p.get("origin_location_id"),
        "Destination": p.get("destination_location_id"),
    }
    for p in lPairs
]

df=pd.DataFrame(rows)
df=pd.DataFrame(rows)
df=pd.DataFrame(rows)
# calc reach 

# basic impl
'''
df["RedReach"] = .5
'''
#more indepth
# Scenario 1, if red is able to fully deny a select 
#Red = ["Medan Air Base", "Phase I & II Recon Deep Staging Location"]

rules = {
    "Medan_Air_Base": [
        {"start": 1, "end": 30, "type": "random", "low": 0.1, "high": 0.8},
        {"start": 31, "end": 44,"type": "const",  "value": 0},
    ],
    "Phase_I__II_Recon_Deep_Staging_Location": [
        {"start": 1, "end": 30, "type": "random", "low": 0.1, "high": 0.8},
        {"start": 31, "end": 44,"type": "const",  "value": 0},
    ],
    # Add more locations here, e.g.:
    # "LOC3": [
    #     {"start": 1, "end": 120, "type": "const", "value": 0},
    # ],
    "Phase_II_D1_PAA_3": [
    	{"start": 1,  "end": 40, "type": "random",  "low": 0.1, "high": 0.8},
        {"start": 41, "end": 44,"type": "const",  "value": 0},
    ],
    "Phase_II_D2_PAA4": [
    	{"start": 1,  "end": 40, "type": "random",  "low": 0.1, "high": 0.8},
        {"start": 41, "end": 44,"type": "const",  "value": 0},
    ],
    "Phase_II_D3_PAA5": [
    	{"start": 1,  "end": 44, "type": "random",  "low": 0.1, "high": 0.8},
    
    ],
    "Phase_II_D1_128__328_Supporting_Position": [
    	{"start": 1,  "end": 40, "type": "random",  "low": 0.1, "high": 0.8},
        {"start": 41, "end": 44,"type": "const",  "value": 0},
    ],
}

num_time_periods = 39

loc_masks = {loc: (df["Destination"] == loc) for loc in rules}

for tp in range(1, num_time_periods + 1):
    # Start with all zeros for this TP
    col = np.zeros(len(df), dtype=float)

    # Apply rules for each location
    for loc, segments in rules.items():
        mask_loc = loc_masks[loc]
        if not mask_loc.any():
            continue  # no rows with this destination, skip

        # Find which segment applies for this TP
        for seg in segments:
            if seg["start"] <= tp <= seg["end"]:
                if seg["type"] == "const":
                    col[mask_loc] = seg["value"]
                elif seg["type"] == "random":
                    low, high = seg["low"], seg["high"]
                    # random values in [low, high) for rows of this location
                    col[mask_loc] = low + np.random.rand(mask_loc.sum()) * (high - low)
                break  # stop after first matching segment

    # Save this TP column to the DataFrame
    df[f"Reach TP{tp}"] = col

df.to_csv("Scenario1.csv",index=False)
print("Success")

