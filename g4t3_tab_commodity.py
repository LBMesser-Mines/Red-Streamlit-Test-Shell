# g4t3_tab_commodity.py
# Renders the content for the 'Commodity Metrics' tab on the Stage Comparison page.

import streamlit as st
import pandas as pd
import g4t3_analysis
import g4t3_visualizations
from g4t3_utils import stage_label

def _aggregate_link_weights_local(rows):
    """
    Aggregates total weight per route for a list of shipment records.
    """
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["weight"] = df["Mvalue"] * df["per-item weight"]
    df["route"] = df["l"] + " → " + df["lprime"]
    return df.groupby("route")["weight"].sum()

def render(all_data, all_routes, baseline_key, compare_key):
    """Renders the content for the Commodity Metrics tab."""
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

    # --- Section 1: Link Weight Analysis ---
    st.title("Link Weight Analysis")
    
    df_base_full = pd.DataFrame(all_data[baseline_key])
    if df_base_full.empty or 't' not in df_base_full.columns:
        st.warning("Baseline data is empty or invalid for link weight analysis.")
        return
        
    t_min = int(df_base_full['t'].min())
    t_max = int(df_base_full['t'].max())
    t_selected = st.slider("Select Time index (t) for Link Analysis", t_min, t_max, t_min, key="comm_time_slider")

    baseline_rows_t = [r for r in all_data[baseline_key] if int(r['t']) == t_selected]
    compare_rows_t = [r for r in all_data.get(compare_key, []) if int(r.get('t', -1)) == t_selected]

    st.subheader(f"Links with the Most Weight @ t={t_selected}")
    st.markdown("This chart shows the top 10 links with the highest total shipment weight from **either** the baseline or comparison stage.")

    base_weights = _aggregate_link_weights_local(baseline_rows_t)
    comp_weights = _aggregate_link_weights_local(compare_rows_t)

    if base_weights.empty and comp_weights.empty:
        st.info(f"No shipment data for either stage at t={t_selected}.")
    else:
        # Get top 10 routes from each stage
        top_10_base = base_weights.nlargest(10)
        top_10_comp = comp_weights.nlargest(10)

        # Combine the indices (routes) into a unique set
        combined_routes = top_10_base.index.union(top_10_comp.index)

        # Create the combined DataFrame using the full set of routes
        combined_df = pd.DataFrame({
            'baseline_weight': base_weights.reindex(combined_routes).fillna(0),
            'compare_weight': comp_weights.reindex(combined_routes).fillna(0)
        })

        # Determine the sort order based on the max weight in either stage
        combined_df['max_weight'] = combined_df[['baseline_weight', 'compare_weight']].max(axis=1)
        combined_df = combined_df.sort_values('max_weight', ascending=False).drop(columns=['max_weight'])
        
        # Reset index to make 'route' a column
        combined_df = combined_df.reset_index().rename(columns={'index': 'route'})

        # Melt the DataFrame for plotting
        plot_df = combined_df.melt(
            id_vars=['route'],
            value_vars=['baseline_weight', 'compare_weight'],
            var_name='stage_type',
            value_name='weight'
        )
        plot_df['stage'] = plot_df['stage_type'].map({
            'baseline_weight': baseline_label,
            'compare_weight': compare_label
        })

        # Define the order for the x-axis based on the new sorting
        route_order = combined_df['route'].tolist()
        plot_df['route'] = pd.Categorical(plot_df['route'], categories=route_order, ordered=True)
        y_max = plot_df['weight'].max() * 1.1 if not plot_df['weight'].empty else 1

        chart = g4t3_visualizations.create_comparison_grouped_bar_chart(
            plot_df, y_max, stage_order
        )
        st.altair_chart(chart, use_container_width=True)


    st.markdown("---")

    # --- Section 3: Peak Commodity Class Weight ---
    st.subheader("Peak Commodity Class Weight")
    st.markdown("The maximum weight of a single shipment on any route for specific commodity classes.")
    
    col3, col4 = st.columns(2)
    classes_to_check = ['I', 'VII']

    with col3:
        st.markdown(f"**{baseline_label}**")
        peak_weights_base = g4t3_analysis.find_max_weight_by_class(all_data[baseline_key], classes_to_check)
        if not peak_weights_base.empty:
            for _, row in peak_weights_base.iterrows():
                st.markdown(f"""
                - **Class {row['class']}**: <span class='purple-text'>{row['weight']:,.0f}</span> lbs
                    - Route: {row['route']}
                    - Time: t={row['t']}
                """, unsafe_allow_html=True)
        else:
            st.info("No Class I or VII shipment data available.")

    with col4:
        st.markdown(f"**{compare_label}**")
        peak_weights_comp = g4t3_analysis.find_max_weight_by_class(all_data[compare_key], classes_to_check)
        if not peak_weights_comp.empty:
            for _, row in peak_weights_comp.iterrows():
                st.markdown(f"""
                - **Class {row['class']}**: <span class='purple-text'>{row['weight']:,.0f}</span> lbs
                    - Route: {row['route']}
                    - Time: t={row['t']}
                """, unsafe_allow_html=True)
        else:
            st.info("No Class I or VII shipment data available.")