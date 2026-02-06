# g4t3_data_loader.py
# Contains functions for loading and preprocessing data from source files.

import json
import glob
import pandas as pd
from collections import Counter

def load_shipment_data():
    """
    Loads all MActivity*.csv files, joins them with location coordinates
    and commodity class information from a JSON file, and returns a 
    dictionary of dataframes.
    """
    try:
        with open("lstx-clb-2Phases.json") as f:
            alaska = json.load(f)
    except FileNotFoundError:
        print("Error: lstx-clb-2Phases.json not found. Cannot join coordinates.")
        return {}
        
    coord_map = {loc["id"]: (loc["latitude"], loc["longitude"]) for loc in alaska["locations"]}
    class_map = {comm["id"]: comm.get("class") for comm in alaska.get("commodities", [])}

    shipments = {}
    for path in glob.glob("MActivity*.csv"):
        df = pd.read_csv(path)
        df.rename(columns=lambda c: c.strip(), inplace=True)
        
        idx = path[len("MActivity"):-4]
        key = f"M{idx}"

        # Join coordinates
        if len(df.columns) >= 2:
            src, dst = df.columns[0], df.columns[1]
            df[src] = df[src].astype(str).str.strip()
            df[dst] = df[dst].astype(str).str.strip()
            df["l_lat"], df["l_long"] = zip(*df[src].map(lambda x: coord_map.get(x, (None, None))))
            df["lprime_lat"], df["lprime_long"] = zip(*df[dst].map(lambda x: coord_map.get(x, (None, None))))
        
        # Join commodity class
        if 'j' in df.columns:
            df['j'] = df['j'].astype(str).str.strip()
            df['class'] = df['j'].map(class_map)

        shipments[key] = df.to_dict(orient="records")

    return shipments

def load_inventory_data():
    """
    Loads all IActivity*.csv files, joins them with commodity class
    information, and returns a dictionary mapping scenario keys to DataFrames.
    """
    try:
        with open("lstx-clb-2Phases.json") as f:
            alaska = json.load(f)
    except FileNotFoundError:
        print("Error: lstx-clb-2Phases.json not found. Cannot join commodity class.")
        return {}
        
    class_map = {str(comm["id"]): comm.get("class") for comm in alaska.get("commodities", [])}

    inv = {}
    for path in glob.glob("IActivity*.csv"):
        # Corrected Logic: Ensure the 'j' column is read as a string from the start
        df = pd.read_csv(path, dtype={'j': str})
        df.rename(columns=lambda c: c.strip(), inplace=True)
        
        # Join commodity class
        if 'j' in df.columns:
            df['j'] = df['j'].str.strip()
            df['class'] = df['j'].map(class_map)

        idx = path[len("IActivity"):-4]
        inv[f"I{idx}"] = df
    return inv

def load_vehicle_usage_data():
    """
    Loads all ZActivity*.csv files into a dictionary mapping scenario keys
    (e.g., "Z0", "Z1") to their corresponding pandas DataFrame.
    """
    vehicles = {}
    for path in glob.glob("ZActivity*.csv"):
        df = pd.read_csv(path)
        df.rename(columns=lambda c: c.strip(), inplace=True)
        idx = path[len("ZActivity"):-4]
        vehicles[f"Z{idx}"] = df
    return vehicles

def load_connector_supply():
    """
    Loads connector supply information from the JSON file, parsing connector IDs
    to count the total supply of each connector type.
    """
    try:
        with open("lstx-clb-2Phases.json") as f:
            alaska = json.load(f)
    except FileNotFoundError:
        print("Error: lstx-clb-2Phases.json not found. Cannot load connector supply.")
        return {}

    connector_types = []
    for connector in alaska.get("connectors", []):
        try:
            parts = connector["id"].split("_", 2)
            if len(parts) > 2:
                connector_types.append(parts[1])
        except (IndexError, TypeError):
            print(f"Warning: Could not parse connector ID: {connector.get('id')}")
            continue
            
    return Counter(connector_types)

def load_all_possible_routes():
    """
    Loads all location IDs from the JSON and creates a DataFrame of all possible
    unique routes (l, lprime) where l != lprime.
    """
    try:
        with open("lstx-clb-2Phases.json") as f:
            alaska = json.load(f)
    except FileNotFoundError:
        print("Error: lstx-clb-2Phases.json not found.")
        return pd.DataFrame()

    location_ids = [loc["id"] for loc in alaska.get("locations", [])]
    
    all_routes = []
    for l in location_ids:
        for lprime in location_ids:
            if l != lprime:
                all_routes.append({'l': l, 'lprime': lprime})
    
    return pd.DataFrame(all_routes)
