# g4t3_analysis.py
# This module contains all core data processing and statistical analysis functions.

import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import pandas as pd

# --- Flow Balance (Coefficient of Variation) Analysis ---

def calculate_cv_per_time(rows: List[dict]) -> pd.DataFrame:
    """
    Calculates the Coefficient of Variation (CV) for flows at each time period,
    considering only active edges (those with non-zero flow).
    CV = Standard Deviation / Mean. A lower CV indicates a more balanced flow.

    Args:
        rows: A list of shipment records for a single stage.

    Returns:
        A DataFrame with columns ['t', 'cv'].
    """
    if not rows:
        return pd.DataFrame(columns=['t', 'cv'])

    df = pd.DataFrame(rows)
    df['weight'] = df['Mvalue'] * df['per-item weight']
    df['route'] = df['l'] + " → " + df['lprime']
    
    active_flows = df[df['weight'] > 0].copy()

    if active_flows.empty:
        return pd.DataFrame(columns=['t', 'cv'])

    flow_per_link = active_flows.groupby(['t', 'route'])['weight'].sum().reset_index()

    cv_per_t = flow_per_link.groupby('t')['weight'].apply(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 0.0
    ).reset_index(name='cv')

    return cv_per_t


def calculate_overall_average_cv(cv_df: pd.DataFrame) -> float:
    """
    Calculates the single average of the 'Coefficient of Variation' across all time periods.
    """
    if cv_df is None or cv_df.empty:
        return 0.0
    return cv_df['cv'].mean()

def calculate_max_link_weight(rows: List[dict]) -> Tuple[float, int, str, str]:
    """
    Calculates the maximum weight found on any single link (route) at any
    single point in time, and returns the details of that link.
    """
    if not rows:
        return (0.0, None, None, None)

    df = pd.DataFrame(rows)
    if df.empty or 'Mvalue' not in df.columns or 'per-item weight' not in df.columns:
        return (0.0, None, None, None)

    df['weight'] = df['Mvalue'] * df['per-item weight']

    # Group by time and route (link), then sum the weights
    link_weights = df.groupby(['t', 'l', 'lprime'])['weight'].sum()

    if link_weights.empty:
        return (0.0, None, None, None)

    # Find the maximum value and its corresponding index (t, l, lprime)
    max_weight_value = link_weights.max()
    t, l, lprime = link_weights.idxmax()

    return (max_weight_value, t, l, lprime)


# --- Vehicle Utilization Analysis ---

def find_max_connector_usage(vehicle_usage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the single highest usage instance for each connector type.
    """
    if vehicle_usage_df is None or vehicle_usage_df.empty:
        return pd.DataFrame()
    
    max_indices = vehicle_usage_df.groupby('h')['Zvalue'].idxmax()
    max_usage_df = vehicle_usage_df.loc[max_indices]
    
    max_usage_df['route'] = max_usage_df['l'] + " → " + max_usage_df['lprime']
    
    return max_usage_df[['h', 'Zvalue', 'route', 't']].sort_values(by='h').reset_index(drop=True)

def calculate_vehicle_utilization(vehicle_usage_df: pd.DataFrame, connector_supply: dict) -> pd.DataFrame:
    """
    Calculates the utilization percentage for each vehicle type over time.
    """
    if vehicle_usage_df.empty or not connector_supply:
        return pd.DataFrame()
    usage_by_time = vehicle_usage_df.groupby(['h', 't'])['Zvalue'].sum().reset_index()
    usage_by_time.rename(columns={'Zvalue': 'in_use'}, inplace=True)
    usage_by_time['supply'] = usage_by_time['h'].map(connector_supply)
    usage_by_time['utilization'] = usage_by_time.apply(
        lambda row: row['in_use'] / row['supply'] if row['supply'] > 0 else 0,
        axis=1
    )
    usage_by_time['utilization'] = usage_by_time['utilization'].clip(upper=1.0)
    return usage_by_time

def calculate_overall_average_utilization(vehicle_usage_df: pd.DataFrame, connector_supply: dict) -> float:
    """
    Calculates the single average utilization metric across all connectors and time periods.
    """
    if vehicle_usage_df is None or vehicle_usage_df.empty or not connector_supply:
        return 0.0
    total_vehicles_used = vehicle_usage_df['Zvalue'].sum()
    total_supply = sum(connector_supply.values())
    num_time_periods = vehicle_usage_df['t'].nunique()
    if total_supply == 0 or num_time_periods == 0:
        return 0.0
    total_possible_usage = total_supply * num_time_periods
    return total_vehicles_used / total_possible_usage if total_possible_usage > 0 else 0.0

# --- Commodity Class Analysis ---

def find_max_weight_by_class(shipment_rows: List[dict], classes_to_find: List[str]) -> pd.DataFrame:
    """
    For each specified commodity class, finds the route and time with the
    single highest total shipment weight for that class by summing all
    commodities of that class on each link.
    """
    if not shipment_rows:
        return pd.DataFrame()

    df = pd.DataFrame(shipment_rows)
    if 'class' not in df.columns or df.empty:
        return pd.DataFrame()

    df_filtered = df[df['class'].isin(classes_to_find)].copy()
    if df_filtered.empty:
        return pd.DataFrame()

    df_filtered['weight'] = df_filtered['Mvalue'] * df['per-item weight']
    df_filtered['route'] = df_filtered['l'] + " → " + df_filtered['lprime']

    summed_weights = df_filtered.groupby(['class', 'route', 't'])['weight'].sum().reset_index()

    max_indices = summed_weights.groupby('class')['weight'].idxmax()
    max_weight_df = summed_weights.loc[max_indices]
    
    return max_weight_df[['class', 'weight', 'route', 't']].sort_values(by='class').reset_index(drop=True)

# --- Temporal Entropy Analysis ---

def compute_temporal_entropy(edges: list, total_intervals: int, log_base=math.e) -> Tuple[float, float, float]:
    """Computes Shannon-style temporal entropy H_t."""
    time_sums = Counter()
    total_weight = sum(edge[3] for edge in edges)
    for edge in edges:
        time_sums[edge[-1]] += edge[3]
    if total_weight <= 0 or total_intervals <= 0: return 0.0, 0.0, 0.0
    h_t = -sum((w / total_weight) * math.log((w / total_weight), log_base) for w in time_sums.values() if w > 0)
    h_t_max = math.log(total_intervals, log_base) if total_intervals > 1 else 0.0
    deficit = 1.0 - (h_t / h_t_max) if h_t_max > 0 else 0.0
    return h_t, h_t_max, deficit

def analyze_temporal_entropy(all_data: Dict[str, list]) -> Dict[str, float]:
    """Calculates the temporal entropy for each scenario."""
    entropies = {}
    for key, rows in all_data.items():
        if not rows: continue
        edges = [(r["l"], r["lprime"], r["h"], r["Mvalue"], r["l_lat"], r["l_long"], r["lprime_lat"], r["lprime_long"], int(r["t"])) for r in rows]
        times = {e[-1] for e in edges}
        total_intervals = max(times) + 1 if times else 0
        h_t, _, _ = compute_temporal_entropy(edges, total_intervals)
        entropies[key] = h_t
    return entropies

# --- Scenario Summary Statistics ---

def compute_summary_stats_for_scenario(rows: List[dict]):
    """Given shipment rows for one scenario, calculates and returns top-10 aggregations."""
    df = pd.DataFrame(rows)
    df["t"] = df["t"].astype(int)
    df["weight"] = df["Mvalue"] * df["per-item weight"]
    df["route"] = df["l"] + " → " + df["lprime"]
    df["commodity"] = df["j"].astype(str)
    route_grp = df.groupby(["t", "route"], as_index=False).agg(total_weight=("weight", "sum"), total_count=("Mvalue", "sum"))
    comm_grp = df.groupby(["t", "commodity"], as_index=False).agg(total_weight=("weight", "sum"), total_count=("Mvalue", "sum"))
    top_w_route = route_grp.sort_values(["t", "total_weight"], ascending=[True, False]).groupby("t").head(10).reset_index(drop=True)
    top_c_route = route_grp.sort_values(["t", "total_count"], ascending=[True, False]).groupby("t").head(10).reset_index(drop=True)
    top_w_comm = comm_grp.sort_values(["t", "total_weight"], ascending=[True, False]).groupby("t").head(10).reset_index(drop=True)
    top_c_comm = comm_grp.sort_values(["t", "total_count"], ascending=[True, False]).groupby("t").head(10).reset_index(drop=True)
    return top_w_route, top_c_route, top_w_comm, top_c_comm, route_grp, comm_grp

# --- Link Weight Comparison ---

def _aggregate_link_weights(rows: List[dict]) -> pd.Series:
    """Helper to create a Series of total weight per route for one time slice."""
    df = pd.DataFrame(rows)
    if df.empty: return pd.Series(dtype=float)
    df["weight"] = df["Mvalue"] * df["per-item weight"]
    df["route"] = df["l"] + " → " + df["lprime"]
    return df.groupby("route")["weight"].sum()

def compute_link_weight_differences(baseline_rows: List[dict], compare_rows: List[dict]) -> pd.DataFrame:
    """Calculates the difference in total weight per link between two scenarios."""
    base = _aggregate_link_weights(baseline_rows)
    comp = _aggregate_link_weights(compare_rows)
    df = pd.DataFrame({"baseline_weight": base, "compare_weight": comp}).fillna(0)
    df["diff"] = df["compare_weight"] - df["baseline_weight"]
    df["abs_diff"] = df["diff"].abs()
    return df.reset_index().rename(columns={"index": "route"})

# --- Inventory Analysis ---

def _coerce_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to clean and format an inventory DataFrame."""
    df = df.copy()
    if 't' in df.columns:
        df["t"] = df["t"].astype(int)
    if 'shortage' in df.columns:
        df["shortage"] = pd.to_numeric(df["shortage"], errors="coerce").fillna(0.0).clip(lower=0)
    if 'Inv_LB' in df.columns:
        df["Inv_LB"] = pd.to_numeric(df["Inv_LB"], errors="coerce").fillna(0.0)
    return df

def calculate_average_fulfillment_at_t(inv_df: pd.DataFrame, selected_t: int) -> float:
    """
    Calculates the average fulfillment rate across all location/commodity pairs
    at a specific time period.
    """
    if inv_df is None or inv_df.empty or 't' not in inv_df.columns:
        return 1.0

    df = _coerce_inventory(inv_df)
    df_t = df[df['t'] == selected_t]

    if df_t.empty:
        return 1.0

    def fulfillment_ratio(row):
        if row['Inv_LB'] > 0:
            return (row['Inv_LB'] - row['shortage']) / row['Inv_LB']
        return 1.0

    df_t['fulfillment_ratio'] = df_t.apply(fulfillment_ratio, axis=1)
    
    return df_t['fulfillment_ratio'].mean()

def get_unfulfilled_demands_at_t(inv_df: pd.DataFrame, selected_t: int) -> pd.DataFrame:
    """
    Finds all commodities at locations where the inventory is less than
    the required lower bound at a specific time period.
    """
    if inv_df is None or inv_df.empty or 't' not in inv_df.columns:
        return pd.DataFrame()

    df = _coerce_inventory(inv_df)
    df_t = df[df['t'] == selected_t]

    unfulfilled = df_t[df_t['shortage'] > 0].copy()

    if unfulfilled.empty:
        return pd.DataFrame()

    unfulfilled['on_hand_inventory'] = unfulfilled['Inv_LB'] - unfulfilled['shortage']

    result_df = unfulfilled[['l', 'j', 'on_hand_inventory', 'Inv_LB']]
    result_df = result_df.rename(columns={
        'l': 'Location',
        'j': 'Commodity',
        'on_hand_inventory': 'On-Hand Inventory',
        'Inv_LB': 'Required Safety Stock'
    })

    # Add the new Fulfillment Rate column
    result_df['Fulfillment Rate'] = result_df.apply(
        lambda row: row['On-Hand Inventory'] / row['Required Safety Stock']
        if row['Required Safety Stock'] > 0 else 1.0,
        axis=1
    )
    
    return result_df.sort_values(by=['Location', 'Commodity']).reset_index(drop=True)

def compute_shortage_fraction_by_commodity(inv_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the overall shortage fraction by commodity across all times/locations."""
    if inv_df.empty: return pd.DataFrame(columns=["commodity", "total_shortage", "total_lb", "ratio"])
    df = _coerce_inventory(inv_df)
    grp = df.groupby("j", as_index=False).agg(total_shortage=("shortage", "sum"), total_lb=("Inv_LB", "sum"))
    grp["ratio"] = grp.apply(lambda r: r["total_shortage"] / r["total_lb"] if r["total_lb"] > 0 else 0.0, axis=1)
    return grp.rename(columns={"j": "commodity"})[["commodity", "total_shortage", "total_lb", "ratio"]]

def calculate_total_demand_fulfilled(inv_df: pd.DataFrame) -> float:
    """
    Calculates the total demand fulfilled as 1 - (sum(shortage) / sum(Inv_LB)).
    This represents a weighted average of fulfillment across all commodities.
    """
    if inv_df is None or inv_df.empty:
        return 1.0

    df = _coerce_inventory(inv_df)
    total_shortage = df['shortage'].sum()
    total_lb = df['Inv_LB'].sum()

    if total_lb == 0:
        return 1.0

    return 1.0 - (total_shortage / total_lb)

def calculate_demand_fulfilled_by_class(inv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the demand fulfilled percentage for each commodity class.
    """
    if inv_df is None or inv_df.empty or 'class' not in inv_df.columns:
        return pd.DataFrame()

    df = _coerce_inventory(inv_df)
    
    class_grp = df.groupby('class').agg(
        total_shortage=('shortage', 'sum'),
        total_lb=('Inv_LB', 'sum')
    ).reset_index()

    class_grp['demand_fulfilled'] = class_grp.apply(
        lambda r: 1 - (r['total_shortage'] / r['total_lb']) if r['total_lb'] > 0 else 1.0,
        axis=1
    )
    
    class_grp.sort_values(by='class', inplace=True)
    
    return class_grp[['class', 'demand_fulfilled']]

def calculate_demand_fulfilled_by_class_location(inv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the demand fulfilled percentage grouped by commodity class and location.
    """
    if inv_df is None or inv_df.empty or 'class' not in inv_df.columns:
        return pd.DataFrame()

    df = _coerce_inventory(inv_df)
    
    class_loc_grp = df.groupby(['class', 'l']).agg(
        total_shortage=('shortage', 'sum'),
        total_lb=('Inv_LB', 'sum')
    ).reset_index()

    class_loc_grp['demand_fulfilled'] = class_loc_grp.apply(
        lambda r: 1 - (r['total_shortage'] / r['total_lb']) if r['total_lb'] > 0 else 1.0,
        axis=1
    )
    
    return class_loc_grp[['class', 'l', 'demand_fulfilled']]

def calculate_inventory_fulfillment(inv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the inventory fulfillment percentage for each location, commodity, and time.
    Fulfillment = On-Hand Inventory / Inventory Lower Bound.
    """
    if inv_df is None or inv_df.empty:
        return pd.DataFrame()
    
    df = _coerce_inventory(inv_df.copy())
    
    df['on-hand inventory'] = df['Inv_LB'] - df['shortage']
    
    df['fulfillment'] = df.apply(
        lambda row: row['on-hand inventory'] / row['Inv_LB'] if row['Inv_LB'] > 0 else 1.0, # Assume 100% if nothing is needed
        axis=1
    )
    
    df.rename(columns={'j': 'commodity'}, inplace=True)
    return df[['l', 't', 'commodity', 'fulfillment']]

def calculate_proportional_shortage_by_location(inv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the proportional contribution of each commodity to the total shortage
    for each location and time period.
    """
    if inv_df is None or inv_df.empty:
        return pd.DataFrame()

    df = _coerce_inventory(inv_df.copy())
    df = df[df['shortage'] > 0]
    if df.empty:
        return pd.DataFrame()

    df['total_shortage_loc_time'] = df.groupby(['l', 't'])['shortage'].transform('sum')
    df['shortage_proportion'] = df.apply(
        lambda row: row['shortage'] / row['total_shortage_loc_time'] if row['total_shortage_loc_time'] > 0 else 0,
        axis=1
    )
    df.rename(columns={'j': 'commodity'}, inplace=True)
    return df[['l', 't', 'commodity', 'shortage_proportion', 'shortage']]