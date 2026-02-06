# g4t3_visualizations.py
# This module contains all functions for creating plots and maps (Altair, Folium).

import folium
import altair as alt
import pandas as pd
from branca.element import Element
from branca.colormap import linear
from collections import defaultdict

# --- Inventory Fulfillment Line Chart ---

def create_fulfillment_line_chart(combined_df: pd.DataFrame, stage_order: list) -> alt.Chart:
    """
    Creates a faceted line chart to compare inventory fulfillment for a specific
    commodity by location over time.

    Args:
        combined_df: A DataFrame with columns ['l', 't', 'stage', 'fulfillment'].
        stage_order: A list of the two stage names for consistent coloring.

    Returns:
        An Altair Chart object.
    """
    if combined_df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="No data available.")

    chart = alt.Chart(combined_df).mark_line(point=True).encode(
        x=alt.X('t:O', title='Time Period'),
        y=alt.Y('fulfillment:Q', title='Fulfillment', axis=alt.Axis(format='%')),
        color=alt.Color('stage:N',
                        scale=alt.Scale(domain=stage_order, range=["#1565c0", "#556B2F"]),
                        legend=alt.Legend(title="Stage")),
        row=alt.Row('l:N', title=None, header=alt.Header(labelAngle=0, labelAlign='left')),
        tooltip=[
            alt.Tooltip('l', title='Location'),
            alt.Tooltip('t', title='Time'),
            alt.Tooltip('stage', title='Stage'),
            alt.Tooltip('fulfillment', title='Fulfillment', format='.1%')
        ]
    ).properties(
        height=150,
        title="Inventory Fulfillment by Location"
    )
    
    return chart

# --- Proportional Shortage Stacked Area Chart ---

def create_proportional_shortage_chart(proportional_df: pd.DataFrame) -> alt.Chart:
    """
    Creates a faceted, normalized stacked area chart to show the proportion
    of total shortage attributed to each commodity, for each location.
    """
    if proportional_df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="No shortage data to display.")
    return alt.Chart(proportional_df).mark_area().encode(
        x=alt.X('t:O', title='Time Period'),
        y=alt.Y('shortage_proportion:Q', stack='normalize', title='Proportion of Shortage', axis=alt.Axis(format='%')),
        color=alt.Color('commodity:N', legend=alt.Legend(title="Commodity")),
        row=alt.Row('l:N', title=None, header=alt.Header(labelAngle=0, labelAlign='left')),
        tooltip=[alt.Tooltip('l', title='Location'), alt.Tooltip('t', title='Time'), alt.Tooltip('commodity', title='Commodity'), alt.Tooltip('shortage', title='Shortage Amount', format=',.0f'), alt.Tooltip('shortage_proportion', title='Proportion', format='.1%')]
    ).properties(height=100, width='container', title="Proportional Commodity Shortage by Location")

# --- Coefficient of Variation Dumbbell Plot ---

def create_cv_dumbbell_plot(combined_df: pd.DataFrame, stage_order: list) -> alt.Chart:
    """
    Creates a dumbbell plot to compare the Coefficient of Variation (CV) over time
    between two scenarios.
    """
    if combined_df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="No data available.")
    line = alt.Chart(combined_df).mark_line(color='gray').encode(
        x=alt.X('t:O', title='Time Period'),
        y=alt.Y('cv:Q', title='Coefficient of Variation'),
        detail='t:O'
    )
    points = alt.Chart(combined_df).mark_point(size=100, filled=True).encode(
        x=alt.X('t:O'),
        y=alt.Y('cv:Q'),
        color=alt.Color('stage:N',
                        scale=alt.Scale(domain=stage_order, range=["#1565c0", "#556B2F"]),
                        legend=alt.Legend(title="Stage")),
        tooltip=[alt.Tooltip('t', title='Time'), alt.Tooltip('stage', title='Stage'), alt.Tooltip('cv', title='CV', format='.2f')]
    )
    return (line + points).properties(height=400, width='container', title="Flow Distribution (Coefficient of Variation)")


# --- Combined Vehicle Utilization Area Chart ---

def create_comparison_utilization_chart(utilization_df: pd.DataFrame, stage_order: list) -> alt.Chart:
    """
    Creates a layered area chart to directly compare vehicle utilization
    between two scenarios for each connector type.
    """
    if utilization_df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="No vehicle data available.")
    return alt.Chart(utilization_df).mark_area(
        opacity=0.6, interpolate='monotone'
    ).encode(
        x=alt.X('t:O', title='Time Period', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('utilization:Q', title='Utilization', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1]), stack=None),
        color=alt.Color('stage:N', scale=alt.Scale(domain=stage_order, range=["#1565c0", "#556B2F"]), legend=alt.Legend(title="Stage")),
        row=alt.Row('h:N', title=None, header=alt.Header(labelAngle=0, labelAlign='left')),
        tooltip=[alt.Tooltip('h', title='Connector'), alt.Tooltip('stage', title='Stage'), alt.Tooltip('t', title='Time'), alt.Tooltip('utilization', title='Utilization', format='.1%')]
    ).properties(height=150, title="Vehicle Utilization by Connector Type")

# --- New Chart: Combined Vehicle Usage Count Area Chart ---

def create_comparison_usage_count_chart(usage_df: pd.DataFrame, stage_order: list) -> alt.Chart:
    """
    Creates a layered area chart to directly compare vehicle usage counts
    between two scenarios for each connector type.
    """
    if usage_df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="No vehicle data available.")
        
    return alt.Chart(usage_df).mark_area(
        opacity=0.6, interpolate='monotone'
    ).encode(
        x=alt.X('t:O', title='Time Period', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('in_use:Q', title='Number of Vehicles Used', stack=None),
        color=alt.Color('stage:N', scale=alt.Scale(domain=stage_order, range=["#1565c0", "#556B2F"]), legend=alt.Legend(title="Stage")),
        row=alt.Row('h:N', title=None, header=alt.Header(labelAngle=0, labelAlign='left')),
        tooltip=[alt.Tooltip('h', title='Connector'), alt.Tooltip('stage', title='Stage'), alt.Tooltip('t', title='Time'), alt.Tooltip('in_use', title='Vehicles Used', format=',.0f')]
    ).properties(height=150, title="Vehicle Usage Count by Connector Type")

# --- Vehicle Utilization Heatmap ---

def create_utilization_heatmap(utilization_df: pd.DataFrame, title: str) -> alt.Chart:
    """Creates an Altair heatmap to visualize vehicle utilization percentage."""
    if utilization_df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="No vehicle data available.")
    return alt.Chart(utilization_df).mark_rect().encode(
        x=alt.X('t:O', title='Time Period'),
        y=alt.Y('h:N', title='Connector Type'),
        color=alt.Color('utilization:Q', title='Utilization', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(format=".0%")),
        tooltip=[alt.Tooltip('h', title='Connector'), alt.Tooltip('t', title='Time'), alt.Tooltip('in_use', title='In Use'), alt.Tooltip('supply', title='Supplied'), alt.Tooltip('utilization', title='Utilization', format='.1%')]
    ).properties(title=title, height=400)

# --- Folium Map Visualizations ---

def create_route_map(df_t: pd.DataFrame, global_min: float, red_threshold: float, unit_label: str = " lbs") -> folium.Map:
    """Creates a Folium map visualizing shipment routes for a specific time index."""
    lats = pd.concat([df_t["l_lat"], df_t["lprime_lat"]]).dropna().tolist()
    lons = pd.concat([df_t["l_long"], df_t["lprime_long"]]).dropna().tolist()
    if not lats or not lons: return folium.Map(location=[61.2, -149.9], zoom_start=4)
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    m = folium.Map(location=center, zoom_start=5)
    css = """
    <style>
      .leaflet-popup-content-wrapper {max-width:140px !important;}
      .leaflet-popup-content {font-size:12px !important;}
      .legend.leaflet-control text { display: none; }
    </style>
    """
    m.get_root().header.add_child(Element(css))
    coords = {}
    for _, r in df_t.iterrows():
        coords[r["l"]] = (r["l_lat"], r["l_long"])
        coords[r["lprime"]] = (r["lprime_lat"], r["lprime_long"])
    for loc, (lat, lon) in coords.items():
        if lat is not None and lon is not None: folium.Marker([lat, lon], popup=loc, tooltip=loc).add_to(m)
    mass_sum = defaultdict(float)
    for _, r in df_t.iterrows():
        mass_sum[(r["l"], r["lprime"])] += r["Mvalue"] * r["per-item weight"]
    if not mass_sum: return m
    mn, mx = global_min, max(red_threshold, global_min + 1)
    cmap = linear.YlOrRd_09.scale(mn, mx)
    cmap.caption = "Risk Index"
    cmap.tick_labels = None
    cmap.width = 120
    cmap.add_to(m)
    for (o, d), total_mass in mass_sum.items():
        if o not in coords or d not in coords: continue
        o_lat, o_lon = coords[o]
        d_lat, d_lon = coords[d]
        if any(v is None for v in [o_lat, o_lon, d_lat, d_lon]): continue
        line_weight = 1 + (total_mass - mn) / (mx - mn) * 7
        glow_weight = line_weight + ((total_mass - mn) / (mx - mn) * 12)
        folium.PolyLine([(o_lat, o_lon), (d_lat, d_lon)], weight=glow_weight, color="white", opacity=0.4).add_to(m)
        color = cmap(min(total_mass, mx))
        html = f"<b>{o} → {d}</b><br>ΣMass: {total_mass:.1f}{unit_label}"
        popup = folium.Popup(folium.IFrame(html, width=140, height=50), max_width=160)
        folium.PolyLine([(o_lat, o_lon), (d_lat, d_lon)], weight=line_weight, color=color, opacity=0.8, popup=popup, tooltip=f"{o} → {d}: {total_mass:.0f}{unit_label}").add_to(m)
    if coords: m.fit_bounds(list(coords.values()), padding=(20, 20))
    return m

# --- Altair Chart Visualizations ---

def create_entropy_chart(df_ent: pd.DataFrame) -> alt.Chart:
    """Creates a line chart for temporal entropy across scenarios."""
    mn, mx = df_ent["Entropy"].min(), df_ent["Entropy"].max()
    padding = (mx - mn) * 0.1
    return alt.Chart(df_ent).mark_line(point=True).encode(
        x=alt.X("Red Iter:O", axis=alt.Axis(title="Red Iteration")),
        y=alt.Y("Entropy:Q", axis=alt.Axis(title="Temporal Entropy"), scale=alt.Scale(domain=[mn - padding, mx + padding])),
    ).properties(height=300)

def create_drilldown_chart(df: pd.DataFrame, y_col: str, y_title: str, y_max: float, group_col: str, group_title: str, title: str, time_domain: list) -> alt.Chart:
    """Generic factory for creating drill-down bar charts."""
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("t:O", title="Time index (t)", scale=alt.Scale(domain=time_domain), axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f"{y_col}:Q", title=y_title, scale=alt.Scale(domain=[0, y_max])),
        xOffset=f"{group_col}:N",
        color=alt.Color(f"{group_col}:N", legend=alt.Legend(title=group_title)),
        tooltip=["t", group_col, y_col],
    ).properties(height=350, title=title)

def create_shortage_chart(df_comm: pd.DataFrame) -> alt.Chart:
    """Creates a bar chart for inventory shortage fraction by commodity."""
    return alt.Chart(df_comm).mark_bar().encode(
        x=alt.X("commodity:N", title="Commodity", sort="-y"),
        y=alt.Y("ratio:Q", title="Shortage Fraction", axis=alt.Axis(format="%")),
        tooltip=["commodity", "total_shortage", "total_lb", "ratio"],
    ).properties(height=400, title="Inventory Shortage Fraction by Commodity")

def create_demand_fulfilled_chart(df_comm: pd.DataFrame) -> alt.Chart:
    """Creates a bar chart for inventory demand fulfilled by commodity."""
    return alt.Chart(df_comm).mark_bar().encode(
        x=alt.X("commodity:N", title="Commodity", sort="-y"),
        y=alt.Y("demand_fulfilled:Q", title="Demand Fulfilled", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["commodity", "total_shortage", "total_lb", "demand_fulfilled"],
    ).properties(height=400, title="Demand Fulfilled by Commodity")

def create_comparison_grouped_bar_chart(long_df: pd.DataFrame, y_max: float, stage_order: list) -> alt.Chart:
    """Creates a grouped bar chart for comparing link weights between two stages."""
    return alt.Chart(long_df).mark_bar().encode(
        x=alt.X("route:N", sort=None, title="Route"),
        y=alt.Y("weight:Q", title="Total Weight", scale=alt.Scale(domain=[0, y_max])),
        xOffset=alt.XOffset("stage:N", scale=alt.Scale(domain=stage_order)),
        color=alt.Color("stage:N", scale=alt.Scale(domain=stage_order, range=["#1565c0", "#556B2F"]), legend=alt.Legend(title="Stage")),
        tooltip=["route", "stage", "weight"],
    ).properties(height=500, width='container')

def create_comparison_shortage_chart(df_avg: pd.DataFrame, stage_order: list) -> alt.Chart:
    """Creates a bar chart comparing average inventory shortage between two stages."""
    return alt.Chart(df_avg).mark_bar().encode(
        x=alt.X("stage:N", title="Stage", sort=stage_order),
        y=alt.Y("avg_shortage:Q", title="Average Shortage Fraction", axis=alt.Axis(format="%")),
        color=alt.Color("stage:N", scale=alt.Scale(domain=stage_order, range=["#1565c0", "#556B2F"]), legend=alt.Legend(title="Stage")),
        tooltip=["stage", "avg_shortage"],
    ).properties(height=400)