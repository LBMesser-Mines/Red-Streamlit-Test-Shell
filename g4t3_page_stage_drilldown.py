# g4t3_page_stage_drilldown.py
# Renders the 'Stage Drill-Down' page for detailed analysis of a single stage.

import streamlit as st
import pandas as pd
import g4t3_analysis
import g4t3_visualizations
from g4t3_utils import stage_label

def render_page(all_shipment_data, all_inventory_data, all_vehicle_usage, connector_supply):
    """
    Displays detailed metrics and charts for a single selected stage.
    """
    st.header("Stage Drill-Down")

    # --- Sidebar Controls ---
    st.sidebar.header("Drill-Down Controls")
    stage_key = st.sidebar.selectbox(
        "Select Stage",
        options=sorted(all_shipment_data.keys(), key=lambda k: int(k.lstrip("M"))),
        format_func=stage_label
    )
    
    st.subheader(f"Displaying Metrics for: {stage_label(stage_key)}")

    # --- Tab Layout ---
    tab1, tab2, tab3 = st.tabs(["Commodity Metrics", "Inventory Metrics", "Connector Metrics"])

    # --- Commodity Metrics Tab ---
    with tab1:
        st.markdown("#### Flow Distribution (Coefficient of Variation)")
        
        with st.expander("Show Formula"):
            st.latex(r'''
            CV_t = \frac{\sigma_t}{\mu_t}
            ''')
            st.markdown(r"""
            Where:
            - $CV_t$ is the Coefficient of Variation at time $t$.
            - $\sigma_t$ is the standard deviation of total shipment weights on all active routes at time $t$.
            - $\mu_t$ is the mean of total shipment weights on all active routes at time $t$.
            The value shown is the average $CV_t$ over all time periods.
            """)

        cv_df = g4t3_analysis.calculate_cv_per_time(all_shipment_data[stage_key])
        avg_cv = g4t3_analysis.calculate_overall_average_cv(cv_df)
        st.metric(label="Average Flow Distribution", value=f"{avg_cv:.3f}")
        st.markdown("*(Lower is better)*")
        
        if not cv_df.empty:
            st.line_chart(cv_df.rename(columns={'t': 'Time', 'cv': 'CV'}).set_index('Time'))


    # --- Inventory Metrics Tab (Updated) ---
    with tab2:
        inv_key = f"I{int(stage_key.lstrip('M'))}"
        df_inv = all_inventory_data.get(inv_key)

        if df_inv is None or df_inv.empty:
            st.warning("No inventory data available for this stage.")
        else:
            st.markdown("#### Overall Demand Fulfilled")
            st.markdown("The average percentage of required inventory that was successfully fulfilled, averaged across all commodities for all time periods and locations.")
            
            with st.expander("Show Formula"):
                st.latex(r'''
                \text{Demand Fulfilled} = 1 - \text{Average}\left(\frac{\sum_{l,t} \text{Shortage}_{j,l,t}}{\sum_{l,t} \text{Inv\_LB}_{j,l,t}}\right)_{\forall j}
                ''')
                st.markdown("Where the average is taken over all commodities $j$.")

            shortage_df = g4t3_analysis.compute_shortage_fraction_by_commodity(df_inv)
            avg_shortage = 0.0
            if not shortage_df.empty:
                avg_shortage = shortage_df['ratio'].mean()
            
            demand_fulfilled = 1 - avg_shortage
            st.metric(label=f"Overall Demand Fulfilled", value=f"{demand_fulfilled:.2%}")

            st.markdown("---")
            
            st.markdown("#### Demand Fulfilled by Commodity")
            st.markdown("The percentage of demand fulfilled for each individual commodity, calculated by summing the total shortage and total demand across all locations and time periods.")

            if not shortage_df.empty:
                shortage_df['demand_fulfilled'] = 1 - shortage_df['ratio']
                chart = g4t3_visualizations.create_demand_fulfilled_chart(shortage_df)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No shortage data to create the chart.")


    # --- Connector Metrics Tab ---
    with tab3:
        vehicle_key = f"Z{int(stage_key.lstrip('M'))}"
        df_vehicle = all_vehicle_usage.get(vehicle_key)

        if df_vehicle is None or df_vehicle.empty:
            st.warning("No vehicle data available for this stage.")
        else:
            st.markdown("#### Average Connector Utilization")
            
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

            avg_util = g4t3_analysis.calculate_overall_average_utilization(df_vehicle, connector_supply)
            st.metric(label="Average Connector Utilization", value=f"{avg_util:.2%}")

            utilization_df = g4t3_analysis.calculate_vehicle_utilization(df_vehicle, connector_supply)
            if not utilization_df.empty:
                heatmap = g4t3_visualizations.create_utilization_heatmap(
                    utilization_df, 
                    title="Connector Utilization Heatmap"
                )
                st.altair_chart(heatmap, use_container_width=True)
