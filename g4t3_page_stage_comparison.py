# g4t3_page_stage_comparison.py
# Renders the main 'Stage Comparison' page and orchestrates the tab layout.

import streamlit as st
import pandas as pd
from g4t3_utils import stage_label
import g4t3_tab_commodity
import g4t3_tab_inventory
import g4t3_tab_connector

def render_page(all_data, inv_data, vehicle_data, connector_supply, all_routes):
    """
    Displays a comparison between two selected stages, using a tabbed interface
    for different metric categories.
    """
    st.header("Stage Comparison")

    # --- Sidebar Controls (remains on the main page) ---
    st.sidebar.header("Comparison Controls")
    all_keys = sorted(all_data.keys(), key=lambda k: int(k.lstrip("M")))
    
    baseline_key = st.sidebar.selectbox(
        "Baseline Stage (Select Cornell DC_LOG)",
        options=all_keys, index=0, format_func=stage_label
    )
    compare_key = st.sidebar.selectbox(
        "Comparison Stage",
        options=[k for k in all_keys if k != baseline_key], index=0, format_func=stage_label
    )
    
    # --- Tab Layout ---
    tab1, tab2, tab3 = st.tabs(["Commodity Metrics", "Inventory Metrics", "Connector Metrics"])

    with tab1:
        # Render the content for the Commodity tab
        g4t3_tab_commodity.render(all_data, all_routes, baseline_key, compare_key)

    with tab2:
        # Render the content for the Inventory tab
        g4t3_tab_inventory.render(inv_data, baseline_key, compare_key)

    with tab3:
        # Render the content for the Connector tab
        g4t3_tab_connector.render(vehicle_data, connector_supply, baseline_key, compare_key)
