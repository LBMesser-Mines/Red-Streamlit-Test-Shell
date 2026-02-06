# g4t3_page_adversarial_risk.py
# Renders the 'Adversarial Risk Report' page for a high-level tradeoff analysis.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import g4t3_analysis
from g4t3_utils import stage_label

def render_page(all_data, inv_data, vehicle_data, connector_supply):
    """
    Displays a high-level tradeoff analysis between two selected stages.
    """
    st.header("Adversarial Risk Report")
    st.markdown("This report provides a high-level summary of the tradeoffs between two selected stages across key performance categories.")

    # --- Sidebar Controls ---
    st.sidebar.header("Report Controls")
    all_keys = sorted(all_data.keys(), key=lambda k: int(k.lstrip("M")))
    
    baseline_key = st.sidebar.selectbox(
        "Baseline Stage",
        options=all_keys, index=0, format_func=stage_label, key="risk_baseline"
    )
    compare_key = st.sidebar.selectbox(
        "Comparison Stage",
        options=[k for k in all_keys if k != baseline_key], index=0, format_func=stage_label, key="risk_compare"
    )

    baseline_label, compare_label = stage_label(baseline_key), stage_label(compare_key)

    # --- Data Calculation ---
    
    # 1. Demand Fulfilled
    inv_key_base = f"I{int(baseline_key.lstrip('M'))}"
    df_inv_base = inv_data.get(inv_key_base)
    shortage_df_base = g4t3_analysis.compute_shortage_fraction_by_commodity(df_inv_base)
    avg_shortage_base = shortage_df_base['ratio'].mean() if not shortage_df_base.empty else 0.0
    demand_fulfilled_base = 1 - avg_shortage_base

    inv_key_comp = f"I{int(compare_key.lstrip('M'))}"
    df_inv_comp = inv_data.get(inv_key_comp)
    shortage_df_comp = g4t3_analysis.compute_shortage_fraction_by_commodity(df_inv_comp)
    avg_shortage_comp = shortage_df_comp['ratio'].mean() if not shortage_df_comp.empty else 0.0
    demand_fulfilled_comp = 1 - avg_shortage_comp

    # 2. Connector Utilization
    vehicle_key_base = f"Z{int(baseline_key.lstrip('M'))}"
    df_vehicle_base = vehicle_data.get(vehicle_key_base)
    avg_util_base = g4t3_analysis.calculate_overall_average_utilization(df_vehicle_base, connector_supply)

    vehicle_key_comp = f"Z{int(compare_key.lstrip('M'))}"
    df_vehicle_comp = vehicle_data.get(vehicle_key_comp)
    avg_util_comp = g4t3_analysis.calculate_overall_average_utilization(df_vehicle_comp, connector_supply)

    # 3. Max Link Weight (Single Point of Failure)
    max_weight_base, t_base, l_base, lprime_base = g4t3_analysis.calculate_max_link_weight(all_data[baseline_key])
    max_weight_comp, t_comp, l_comp, lprime_comp = g4t3_analysis.calculate_max_link_weight(all_data[compare_key])

    # --- Percentage Change Calculation (Raw) ---
    demand_fulfilled_change = ((demand_fulfilled_comp - demand_fulfilled_base) / demand_fulfilled_base) * 100 if demand_fulfilled_base != 0 else 0
    connector_util_change = ((avg_util_comp - avg_util_base) / avg_util_base) * 100 if avg_util_base != 0 else 0
    max_weight_change = ((max_weight_comp - max_weight_base) / max_weight_base) * 100 if max_weight_base != 0 else 0

    # --- Chart Visualization with Plotly ---
    st.subheader(f"Performance Change: {compare_label} vs. {baseline_label}")

    # Prepare data for the waterfall chart
    waterfall_data = pd.DataFrame([
        {"metric": "Saved Commodity Weight in Event of a Single Link Attack", "change": -max_weight_change},
        {"metric": "Change in Demand Fulfilled", "change": demand_fulfilled_change},
        {"metric": "Change in Connector Utilization", "change": connector_util_change},
    ])
    
    waterfall_data['is_positive'] = waterfall_data['change'] >= 0
    sorted_waterfall_data = waterfall_data.sort_values(
        by=['is_positive', 'change'],
        ascending=[False, False]
    )
    
    pos_sum = sorted_waterfall_data[sorted_waterfall_data['change'] >= 0]['change'].sum()
    neg_sum = sorted_waterfall_data[sorted_waterfall_data['change'] < 0]['change'].sum()
    y_max_range = pos_sum * 1.15
    y_min_range = neg_sum * 1.15 if neg_sum < 0 else -10 

    measures = ["relative"] * len(sorted_waterfall_data) + ["total"]
    x_labels = list(sorted_waterfall_data['metric']) + ["Net Change"]
    y_values = list(sorted_waterfall_data['change']) + [0]
    
    net_change = sorted_waterfall_data['change'].sum()
    text_labels = [f"{v:+.2f}%" for v in sorted_waterfall_data['change']] + [f"<b>{net_change:+.2f}%</b>"]

    total_color = "#006400" if net_change >= 0 else "#B22222"

    fig = go.Figure(go.Waterfall(
        orientation = "v",
        measure = measures,
        x = x_labels,
        y = y_values,
        text = text_labels,
        textposition = "outside",
        connector = {"line":{"color":"black"}},
        increasing = {"marker":{"color":"#2E8B57"}},
        decreasing = {"marker":{"color":"#B22222"}},
        totals = {"marker":{"color": total_color}}
    ))

    fig.update_layout(
        title = "Performance Change Waterfall",
        yaxis_title="Net % Change",
        showlegend = False,
        xaxis_tickangle=0,
        yaxis_range=[y_min_range, y_max_range]
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Single Point of Failure Display ---
    st.subheader("Single Point of Failure Analysis")
    st.markdown("This metric identifies the maximum weight found on any single link at any single time period, representing the worst-case loss from a single, targeted interdiction.")

    col_spof_1, col_spof_2 = st.columns(2)
    with col_spof_1:
        st.metric(
            label=f"Max Weight on a Single Link ({baseline_label})",
            value=f"{max_weight_base:,.0f} lbs"
        )
        if t_base is not None:
            st.caption(f"Link: {l_base} → {lprime_base} at t={t_base}")
    with col_spof_2:
        st.metric(
            label=f"Max Weight on a Single Link ({compare_label})",
            value=f"{max_weight_comp:,.0f} lbs",
            delta=f"{max_weight_change:.2f}%",
            delta_color="inverse"
        )
        if t_comp is not None:
            st.caption(f"Link: {l_comp} → {lprime_comp} at t={t_comp}")

    # --- Data Reference Table ---
    st.subheader("Metric Details")
    
    data_details = {
        'Metric': ['Overall Demand Fulfilled', 'Avg. Connector Utilization', 'Max Link Weight'],
        baseline_label: [f"{demand_fulfilled_base:.2%}", f"{avg_util_base:.2%}", f"{max_weight_base:,.0f} lbs"],
        compare_label: [f"{demand_fulfilled_comp:.2%}", f"{avg_util_comp:.2%}", f"{max_weight_comp:,.0f} lbs"],
        '% Change': [f"{demand_fulfilled_change:+.2f}%", f"{connector_util_change:+.2f}%", f"{max_weight_change:+.2f}%"]
    }
    df_details = pd.DataFrame(data_details)
    st.table(df_details.set_index('Metric'))