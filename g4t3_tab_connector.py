# g4t3_tab_connector.py
# Renders the content for the 'Connector Metrics' tab on the Stage Comparison page.

import streamlit as st
import pandas as pd
import g4t3_analysis
import g4t3_visualizations
from g4t3_utils import stage_label

def render(vehicle_data, connector_supply, baseline_key, compare_key):
    """
    Renders all charts and metrics related to connector/vehicle utilization.
    """
    # --- Custom CSS for Colors ---
    st.markdown("""
    <style>
    h1 { color: #FFD700; }
    div[data-testid="stMetricValue"] { color: #9370DB; }
    .purple-text { color: #9370DB; }
    </style>
    """, unsafe_allow_html=True)

    baseline_label, compare_label = stage_label(baseline_key), stage_label(compare_key)
    stage_order = [baseline_label, compare_label]
    
    vehicle_key_base = f"Z{int(baseline_key.lstrip('M'))}"
    vehicle_key_comp = f"Z{int(compare_key.lstrip('M'))}"
    
    df_vehicle_base = vehicle_data.get(vehicle_key_base)
    df_vehicle_comp = vehicle_data.get(vehicle_key_comp)

    # --- Prepare combined dataframe for all charts ---
    combined_usage_df = pd.DataFrame()
    if df_vehicle_base is not None:
        # Note: calculate_vehicle_utilization also calculates 'in_use' which we need for the count chart
        usage_base = g4t3_analysis.calculate_vehicle_utilization(df_vehicle_base, connector_supply)
        usage_base['stage'] = baseline_label
        combined_usage_df = pd.concat([combined_usage_df, usage_base])

    if df_vehicle_comp is not None:
        usage_comp = g4t3_analysis.calculate_vehicle_utilization(df_vehicle_comp, connector_supply)
        usage_comp['stage'] = compare_label
        combined_usage_df = pd.concat([combined_usage_df, usage_comp])

    # --- Section 1: Utilization by Percentage ---
    st.title("Connector Metrics")
    st.subheader("Average Connector Utilization")
    st.markdown("The average percentage of supplied vehicles that were in use, across all vehicle types and time periods.")

    with st.expander("Show Formula"):
        st.latex(r'''
        \text{Avg. Utilization} = \frac{\sum_{h,l,l',t} Z_{h,l,l',t}}{(\sum_{h} \text{Supply}_h) \times |T|}
        ''')
        st.markdown(r"""
        Where:
        - $Z_{h,l,l',t}$ is the number of connectors of type $h$ used on a route at time $t$.
        - $\text{Supply}_h$ is the total number of available connectors of type $h$.
        - $|T|$ is the total number of time periods.
        """)

    col1, col2 = st.columns(2)
    
    avg_util_base = g4t3_analysis.calculate_overall_average_utilization(df_vehicle_base, connector_supply)
    avg_util_comp = g4t3_analysis.calculate_overall_average_utilization(df_vehicle_comp, connector_supply)

    with col1:
        st.metric(
            label=f"Avg. Connector Utilization ({baseline_label})",
            value=f"{avg_util_base:.2%}"
        )
    with col2:
        st.metric(
            label=f"Avg. Connector Utilization ({compare_label})",
            value=f"{avg_util_comp:.2%}"
        )

    # Vehicle Utilization Chart (Percentage)
    st.subheader("Vehicle Utilization by Connector Type")
    if not combined_usage_df.empty:
        last_active_t_util = combined_usage_df[combined_usage_df['utilization'] > 0]['t'].max()
        filtered_usage_df_util = combined_usage_df[combined_usage_df['t'] <= last_active_t_util]
        utilization_chart = g4t3_visualizations.create_comparison_utilization_chart(filtered_usage_df_util, stage_order)
        st.altair_chart(utilization_chart, use_container_width=True)
    else:
        st.warning("No vehicle data available for the selected stages.")

    st.markdown("---")

    # --- Section 2: Usage by Count ---
    st.subheader("Peak Connector Usage")
    st.markdown("The maximum number of connectors of a specific type used on a single route at any one time.")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"**{baseline_label}**")
        max_usage_base = g4t3_analysis.find_max_connector_usage(df_vehicle_base)
        if not max_usage_base.empty:
            for _, row in max_usage_base.iterrows():
                st.markdown(f"""
                - **{row['h']}**: <span class='purple-text'>{row['Zvalue']}</span> vehicles
                    - Route: {row['route']}
                    - Time: t={row['t']}
                """, unsafe_allow_html=True)
        else:
            st.info("No vehicle data available.")

    with col4:
        st.markdown(f"**{compare_label}**")
        max_usage_comp = g4t3_analysis.find_max_connector_usage(df_vehicle_comp)
        if not max_usage_comp.empty:
            for _, row in max_usage_comp.iterrows():
                st.markdown(f"""
                - **{row['h']}**: <span class='purple-text'>{row['Zvalue']}</span> vehicles
                    - Route: {row['route']}
                    - Time: t={row['t']}
                """, unsafe_allow_html=True)
        else:
            st.info("No vehicle data available.")
            
    # Vehicle Usage Count Chart (Absolute Number)
    st.subheader("Vehicle Usage Count by Connector Type")
    if not combined_usage_df.empty:
        last_active_t_count = combined_usage_df[combined_usage_df['in_use'] > 0]['t'].max()
        filtered_usage_df_count = combined_usage_df[combined_usage_df['t'] <= last_active_t_count]
        usage_count_chart = g4t3_visualizations.create_comparison_usage_count_chart(filtered_usage_df_count, stage_order)
        st.altair_chart(usage_count_chart, use_container_width=True)

