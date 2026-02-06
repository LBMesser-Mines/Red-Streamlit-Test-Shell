# g4t3_tab_inventory.py
# Renders the content for the 'Inventory Metrics' tab on the Stage Comparison page.

import streamlit as st
import pandas as pd
import g4t3_analysis
import g4t3_visualizations
from g4t3_utils import stage_label

def render(inv_data, baseline_key, compare_key):
    """Renders the content for the Inventory Metrics tab."""
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

    inv_key_base = f"I{int(baseline_key.lstrip('M'))}"
    inv_key_comp = f"I{int(compare_key.lstrip('M'))}"

    df_inv_base = inv_data.get(inv_key_base)
    df_inv_comp = inv_data.get(inv_key_comp)

    # --- Main Title ---
    st.title("Closure")

    # --- Inventory Demand Fulfillment Rate at a Selected Time Period ---
    st.subheader("The Inventory Demand Fulfillment Rate by the end of the Selected Time Period")

    # Determine the full time range from both datasets
    min_t, max_t = 0, 1
    if df_inv_base is not None and not df_inv_base.empty and 't' in df_inv_base.columns:
        min_t = int(df_inv_base['t'].min())
        max_t = int(df_inv_base['t'].max())
    if df_inv_comp is not None and not df_inv_comp.empty and 't' in df_inv_comp.columns:
        min_t = min(min_t, int(df_inv_comp['t'].min()))
        max_t = max(max_t, int(df_inv_comp['t'].max()))

    t_selected_unfulfilled = st.slider(
        "Select Time Period for Analysis",
        min_value=min_t,
        max_value=max_t,
        value=max_t, # Default to the last time period
        key="unfulfilled_time_slider"
    )

    # Display the Average Fulfillment Rate metric
    avg_fulfillment_base = g4t3_analysis.calculate_average_fulfillment_at_t(df_inv_base, t_selected_unfulfilled)
    avg_fulfillment_comp = g4t3_analysis.calculate_average_fulfillment_at_t(df_inv_comp, t_selected_unfulfilled)

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(
            label=f"Inventory Demand Fulfilment ({baseline_label})",
            value=f"{avg_fulfillment_base:.2%}"
        )
    with metric_col2:
        st.metric(
            label=f"Inventory Demand Fulfilment ({compare_label})",
            value=f"{avg_fulfillment_comp:.2%}"
        )

    # Display the table of unfulfilled items
    st.markdown("###### Unfulfilled Items Table")
    col_unfulfilled_1, col_unfulfilled_2 = st.columns(2)

    with col_unfulfilled_1:
        st.markdown(f"**{baseline_label} (t={t_selected_unfulfilled})**")
        unfulfilled_base_df = g4t3_analysis.get_unfulfilled_demands_at_t(df_inv_base, t_selected_unfulfilled)
        if not unfulfilled_base_df.empty:
            st.dataframe(unfulfilled_base_df.style.format({
                "On-Hand Inventory": "{:,.0f}",
                "Required Safety Stock": "{:,.0f}",
                "Fulfillment Rate": "{:.2%}"
            }), use_container_width=True)
        else:
            st.info(f"All demands were met at t={t_selected_unfulfilled}.")

    with col_unfulfilled_2:
        st.markdown(f"**{compare_label} (t={t_selected_unfulfilled})**")
        unfulfilled_comp_df = g4t3_analysis.get_unfulfilled_demands_at_t(df_inv_comp, t_selected_unfulfilled)
        if not unfulfilled_comp_df.empty:
            st.dataframe(unfulfilled_comp_df.style.format({
                "On-Hand Inventory": "{:,.0f}",
                "Required Safety Stock": "{:,.0f}",
                "Fulfillment Rate": "{:.2%}"
            }), use_container_width=True)
        else:
            st.info(f"All demands were met at t={t_selected_unfulfilled}.")

    st.markdown("---")
    
    # --- Inventory Demand Fulfilled (All Time) ---
    st.subheader("Average Inventory Demand Fulfilled (All Time Periods)")
    st.markdown("The average percentage of required inventory that was successfully fulfilled, averaged across all unique commodity types for all time periods and locations. Note that all commodities are equally weighted in this metric.")

    col1, col2 = st.columns(2)

    avg_shortage_base = 0.0
    if df_inv_base is not None and not df_inv_base.empty:
        shortage_df_base = g4t3_analysis.compute_shortage_fraction_by_commodity(df_inv_base)
        if not shortage_df_base.empty:
            avg_shortage_base = shortage_df_base['ratio'].mean()

    avg_shortage_comp = 0.0
    if df_inv_comp is not None and not df_inv_comp.empty:
        shortage_df_comp = g4t3_analysis.compute_shortage_fraction_by_commodity(df_inv_comp)
        if not shortage_df_comp.empty:
            avg_shortage_comp = shortage_df_comp['ratio'].mean()

    demand_fulfilled_base = 1 - avg_shortage_base
    demand_fulfilled_comp = 1 - avg_shortage_comp

    with col1:
        st.metric(
            label=f"Average Demand Fulfilled ({baseline_label})",
            value=f"{demand_fulfilled_base:.2%}"
        )
    with col2:
        st.metric(
            label=f"Average Demand Fulfilled ({compare_label})",
            value=f"{demand_fulfilled_comp:.2%}"
        )
    
    # --- Total Demand Fulfilled ---
    st.subheader("Total Inventory Demand Fulfilled (All Time Periods)")
    st.markdown("The percentage of total demand met across all commodities and locations.")

    total_demand_fulfilled_base = g4t3_analysis.calculate_total_demand_fulfilled(df_inv_base)
    total_demand_fulfilled_comp = g4t3_analysis.calculate_total_demand_fulfilled(df_inv_comp)

    col2a, col2b = st.columns(2)
    with col2a:
        st.metric(
            label=f"Total Demand Fulfilled ({baseline_label})",
            value=f"{total_demand_fulfilled_base:.2%}"
        )
    with col2b:
        st.metric(
            label=f"Total Demand Fulfilled ({compare_label})",
            value=f"{total_demand_fulfilled_comp:.2%}"
        )

    # --- Demand Fulfilled by Class (All Time) ---
    st.subheader("Demand Fulfilled by Commodity Class (All Time Periods)")
    st.markdown("The percentage of total demand met for each commodity class, summed across all locations and time periods.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"**{baseline_label}**")
        df_class_base = g4t3_analysis.calculate_demand_fulfilled_by_class(df_inv_base)
        if not df_class_base.empty:
            for _, row in df_class_base.iterrows():
                st.markdown(f"- **Class {row['class']}**: <span class='purple-text'>{row['demand_fulfilled']:.2%}</span>", unsafe_allow_html=True)
        else:
            st.info("No classified inventory data available.")

    with col4:
        st.markdown(f"**{compare_label}**")
        df_class_comp = g4t3_analysis.calculate_demand_fulfilled_by_class(df_inv_comp)
        if not df_class_comp.empty:
            for _, row in df_class_comp.iterrows():
                st.markdown(f"- **Class {row['class']}**: <span class='purple-text'>{row['demand_fulfilled']:.2%}</span>", unsafe_allow_html=True)
        else:
            st.info("No classified inventory data available.")

    st.markdown("---")
    
    # --- Fulfillment Over Time Chart ---
    st.subheader("Inventory Fulfillment by Location (All Time Periods)")
    st.markdown("A detailed view of inventory fulfillment for a selected commodity across different locations over time. Only locations with less than 100% fulfillment at any time period are shown.")
    
    all_commodities = []
    if df_inv_base is not None and 'j' in df_inv_base.columns:
        all_commodities.extend(df_inv_base['j'].unique())
    if df_inv_comp is not None and 'j' in df_inv_comp.columns:
        all_commodities.extend(df_inv_comp['j'].unique())
    
    if all_commodities:
        selected_commodity = st.selectbox(
            "Select a Commodity to Analyze",
            options=sorted(list(set(all_commodities)))
        )

        combined_fulfillment_df = pd.DataFrame()
        if df_inv_base is not None:
            fulfillment_base = g4t3_analysis.calculate_inventory_fulfillment(df_inv_base)
            fulfillment_base_filtered = fulfillment_base[fulfillment_base['commodity'] == selected_commodity]
            if not fulfillment_base_filtered.empty:
                fulfillment_base_filtered['stage'] = baseline_label
                combined_fulfillment_df = pd.concat([combined_fulfillment_df, fulfillment_base_filtered])

        if df_inv_comp is not None:
            fulfillment_comp = g4t3_analysis.calculate_inventory_fulfillment(df_inv_comp)
            fulfillment_comp_filtered = fulfillment_comp[fulfillment_comp['commodity'] == selected_commodity]
            if not fulfillment_comp_filtered.empty:
                fulfillment_comp_filtered['stage'] = compare_label
                combined_fulfillment_df = pd.concat([combined_fulfillment_df, fulfillment_comp_filtered])

        if not combined_fulfillment_df.empty:
            locations_with_shortfall = combined_fulfillment_df[combined_fulfillment_df['fulfillment'] < 1.0]['l'].unique()
            filtered_chart_df = combined_fulfillment_df[combined_fulfillment_df['l'].isin(locations_with_shortfall)]
            
            if not filtered_chart_df.empty:
                fulfillment_chart = g4t3_visualizations.create_fulfillment_line_chart(filtered_chart_df, stage_order)
                st.altair_chart(fulfillment_chart, use_container_width=True)
            else:
                st.info(f"All locations maintained 100% fulfillment for commodity '{selected_commodity}' in the selected stages.")
        else:
            st.warning(f"No inventory data available for commodity '{selected_commodity}' in the selected stages.")
    
    st.markdown("---")

    # --- Demand Fulfilled by Class and Location Table ---
    st.subheader("Demand Fulfilled by Commodity Class and Location (All Time Periods)")
    st.markdown("The percentage of total demand met for each commodity class at each location, summed across all time periods.")

    df_class_loc_base = g4t3_analysis.calculate_demand_fulfilled_by_class_location(df_inv_base)
    df_class_loc_comp = g4t3_analysis.calculate_demand_fulfilled_by_class_location(df_inv_comp)

    if not df_class_loc_base.empty or not df_class_loc_comp.empty:
        merged_df = pd.merge(
            df_class_loc_base,
            df_class_loc_comp,
            on=['class', 'l'],
            how='outer',
            suffixes=(f'_{baseline_label}', f'_{compare_label}')
        )
        
        merged_df.rename(columns={
            'l': 'Location',
            'class': 'Class',
            f'demand_fulfilled_{baseline_label}': f'{baseline_label} (%)',
            f'demand_fulfilled_{compare_label}': f'{compare_label} (%)'
        }, inplace=True)

        merged_df.fillna('-', inplace=True)
        merged_df.sort_values(by=['Location', 'Class'], inplace=True)

        format_dict = {}
        if f'{baseline_label} (%)' in merged_df.columns:
            format_dict[f'{baseline_label} (%)'] = '{:.2%}'
        if f'{compare_label} (%)' in merged_df.columns:
            format_dict[f'{compare_label} (%)'] = '{:.2%}'
        
        st.dataframe(merged_df.style.format(format_dict), use_container_width=True)

    else:
        st.warning("No inventory data with commodity classes available for this comparison.")