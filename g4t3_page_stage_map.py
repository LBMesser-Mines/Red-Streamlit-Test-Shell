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
    # Add custom CSS to align the columns by removing the top margin from the h6 title
    st.markdown("""
        <style>
            h6 {
                margin-top: 0 !important;
            }
            /* Reduce whitespace around streamlit metric */
            div[data-testid="stMetric"] {
                padding-top: 0px !important;
                padding-bottom: 0px !important;
            }
            .purple-text { color: #9370DB; }
        </style>
    """, unsafe_allow_html=True)

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
        dft = df[df["t"] == t_selected].copy()
        df_t_by_key[key] = dft
        
        if not dft.empty:
            dft['weight'] = dft['Mvalue'] * dft['per-item weight']
            total_weights[key] = dft['weight'].sum()
            
            route_masses = dft.groupby(["l", "lprime"])["weight"].sum()
            if not route_masses.empty:
                global_min_mass = min(global_min_mass, route_masses.min())
                global_max_mass = max(global_max_mass, route_masses.max())

    if global_min_mass == float("inf"):
        st.info(f"No shipment data found at time index t={t_selected}.")
        return

    # --- Pre-calculate Cornell (M0) Max Link Weight for comparison ---
    max_weight_cornell_t = 0
    df_cornell_t = df_t_by_key.get("M0")
    if df_cornell_t is not None and not df_cornell_t.empty:
        link_weights_cornell = df_cornell_t.groupby(['l', 'lprime'])['weight'].sum()
        if not link_weights_cornell.empty:
            max_weight_cornell_t = link_weights_cornell.max()

    # --- More Sidebar Controls (dependent on data) ---
    st.sidebar.subheader("Color Scale")
    red_threshold = st.sidebar.number_input(
        "Red Color Threshold (Max Weight)",
        min_value=float(global_min_mass),
        max_value=float(global_max_mass),
        value=float(global_max_mass),
        step=float((global_max_mass - global_min_mass) / 20 or 1.0),
    )

    # --- Display Maps ---
    cols = st.columns(len(selected_keys))
    for i, key in enumerate(selected_keys):
        with cols[i]:
            st.markdown(f"<h6>{stage_label(key)} @ t={t_selected}</h6>", unsafe_allow_html=True)
            dft = df_t_by_key.get(key)

            if dft is None or dft.empty:
                st.warning("No data at this time index.")
                continue

            folium_map = g4t3_visualizations.create_route_map(dft, global_min_mass, red_threshold, unit_label=" lbs")
            st_folium(folium_map, width="100%", height=350, key=f"map_{key}")
            
            # --- Single Point of Failure Metric ---
            st.markdown("###### Single Link Vulnerability")
            link_weights_t = dft.groupby(['l', 'lprime'])['weight'].sum()
            if not link_weights_t.empty:
                max_weight_current = link_weights_t.max()
                
                delta_val = None
                if key != "M0" and max_weight_cornell_t > 0:
                    delta_val = f"{((max_weight_current - max_weight_cornell_t) / max_weight_cornell_t) * 100:.2f}%"

                st.metric(
                    label="Loss of commodity weight if a single link is attacked",
                    value=f"{max_weight_current:,.0f} lbs",
                    delta=delta_val,
                    delta_color="inverse"
                )
                
                # Add the absolute difference bubble for comparison stages or a placeholder for alignment
                if key != "M0" and max_weight_cornell_t > 0:
                    abs_diff = max_weight_current - max_weight_cornell_t
                    color = "#2E8B57" if abs_diff < 0 else "#B22222"
                    bubble_html = f"""
                    <div style="display: inline-block; padding: 3px 10px; border-radius: 15px; background-color: {color}; color: white; font-size: 0.8em; font-weight: 500; margin-top: 5px;">
                        {abs_diff:+,} lbs
                    </div>
                    """
                    st.markdown(bubble_html, unsafe_allow_html=True)
                else:
                    # Add a placeholder div that takes up the same vertical space as the bubble
                    st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)

            else:
                st.metric(label="Max Weight on Single Link", value="0 lbs")
                st.markdown('<div style="height: 29px;"></div>', unsafe_allow_html=True)

            # Add a larger vertical space before the next section
            st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)

            # --- Top 5 Links by Weight ---
            st.markdown("###### Top 5 Links by Weight")
            top_links = link_weights_t.sort_values(ascending=False).head(5)
            if not top_links.empty:
                for (l, lprime), weight in top_links.items():
                    st.markdown(
                        f"• {l} → {lprime}: <span class='purple-text'>{weight:,.0f} lbs</span>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("No links with weight to display.")

            # --- Total Weight Shipped (Moved to bottom) ---
            st.metric(label="Total Weight Shipped", value=f"{total_weights.get(key, 0):,.0f} lbs")