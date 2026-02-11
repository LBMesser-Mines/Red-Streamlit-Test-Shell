# g4t3_app.py
# Main entry point for the Streamlit application.
# This file handles data loading, page navigation, and orchestrates the app's structure.

import streamlit as st
import g4t3_data_loader
import g4t3_page_welcome
import g4t3_page_adversarial_risk
import g4t3_page_stage_drilldown
import g4t3_page_stage_comparison
import g4t3_page_stage_map

from pathlib import Path
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import MAP2.mapping_tool as mapping_tool

# --- Page Configuration ---
st.set_page_config(page_title="G4T3 Dashboard", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width:100% !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Data Loading ---
@st.cache_data
def load_all_data():
    """Loads all necessary data for the application."""
    shipment_data = g4t3_data_loader.load_shipment_data()
    inventory_data = g4t3_data_loader.load_inventory_data()
    vehicle_usage_data = g4t3_data_loader.load_vehicle_usage_data()
    connector_supply = g4t3_data_loader.load_connector_supply()
    all_possible_routes = g4t3_data_loader.load_all_possible_routes()
    return shipment_data, inventory_data, vehicle_usage_data, connector_supply, all_possible_routes

all_shipment_data, all_inventory_data, all_vehicle_usage, connector_supply, all_possible_routes = load_all_data()

# --- Page Routing ---
PAGES = {
    "Welcome": g4t3_page_welcome.render_welcome_page,
    "How G4T3 Works": g4t3_page_welcome.render_how_it_works_page,
    "Adversarial Risk Report": g4t3_page_adversarial_risk.render_page,
    "Stage Drill-Down": g4t3_page_stage_drilldown.render_page,
    "Stage Comparison": g4t3_page_stage_comparison.render_page,
    "Map": mapping_tool.main,
}

def main():
    """Main function to run the Streamlit app."""
    st.sidebar.title("Menu")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page_function = PAGES[selection]

    # Pass the correct data to each page function
    if selection == "Stage Drill-Down":
        page_function(all_shipment_data, all_inventory_data, all_vehicle_usage, connector_supply)
    elif selection == "Stage Comparison":
        page_function(all_shipment_data, all_inventory_data, all_vehicle_usage, connector_supply, all_possible_routes)
    elif selection == "Adversarial Risk Report":
        page_function(all_shipment_data, all_inventory_data, all_vehicle_usage, connector_supply)
    elif selection == "Stage Map":
        page_function(all_shipment_data)
    else:
        page_function()

if __name__ == "__main__":
    main()