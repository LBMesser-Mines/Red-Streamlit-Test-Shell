# g4t3_page_stage_map.py
# Renders the 'Stage Map' page for visualizing shipment routes.

import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
import g4t3_visualizations
from g4t3_utils import stage_label

def render_page(all_data):
    """
    Displays an interactive map of shipment routes for selected stages and time.
    """
    st.header("Stage Map")

    # --- Sidebar Controls ---
    st.sidebar.header("Map Controls")
    
    sorted_keys = sorted(all_data.keys(), key=lambda k: int(k.lstrip("M")))
    selected_keys = st.sidebar.multiselect(
        "Select Stages",
        options=sorted_keys,
        default=sorted_keys,
        format_func=stage_label,
    )
    if not selected_keys:
        st.warning("Please select at least one stage to display.")
        return

    # Prepare dataframes and find global time range
    dfs = {k: pd.DataFrame(all_data[k]).assign(t=lambda df: df["t"].astype(int)) for k in selected_keys}
    min_times = [df["t"].min() for df in dfs.values() if not df.empty]
    max_times = [df["t"].max() for df in dfs.values() if not df.empty]

    if not min_times or not max_times:
        st.warning("Selected stages contain no data.")
        return
        
    t_min_val, t_max_val = int(min(min_times)), int(max(max_times))
    
    t_selected = st.sidebar.slider(
        "Time index (t)",
        min_value=t_min_val,
        max_value=t_max_val,
        value=t_min_val,
    )

    # --- Data Filtering and Pre-calculation ---
    global_min_mass, global_max_mass = float("inf"), float("-inf")
    df_t_by_key, total_weights = {}, {}

    for key, df in dfs.items():
        dft = df[df["t"] == t_selected]
        df_t_by_key[key] = dft
        
        if not dft.empty:
            dft_mass = dft["Mvalue"] * dft["per-item weight"]
            total_weights[key] = dft_mass.sum()
            
            # Calculate min/max for color scale
            route_masses = dft.assign(mass=dft_mass).groupby(["l", "lprime"])["mass"].sum()
            if not route_masses.empty:
                global_min_mass = min(global_min_mass, route_masses.min())
                global_max_mass = max(global_max_mass, route_masses.max())

    if global_min_mass == float("inf"):
        st.info(f"No shipment data found at time index t={t_selected}.")
        return

    # --- More Sidebar Controls (dependent on data) ---
    st.sidebar.subheader("Color Scale")
    red_threshold = st.sidebar.number_input(
        "Red Color Threshold (Max Weight)",
        min_value=global_min_mass,
        max_value=global_max_mass,
        value=global_max_mass,
        step=(global_max_mass - global_min_mass) / 20 or 1.0,
    )

    # --- Display Maps ---
    cols = st.columns(len(selected_keys))
    for i, key in enumerate(selected_keys):
        with cols[i]:
            st.markdown(f"#### {stage_label(key)} @ t={t_selected}")
            dft = df_t_by_key.get(key)

            if dft is None or dft.empty:
                st.warning("No data at this time index.")
                continue

            # Create and display the map for the current stage
            folium_map = g4t3_visualizations.create_route_map(dft, global_min_mass, red_threshold)
            st_folium(folium_map, width="100%", height=350, key=f"map_{key}")
            
            st.metric(label="Total Weight Shipped", value=f"{total_weights.get(key, 0):,.0f} t")
