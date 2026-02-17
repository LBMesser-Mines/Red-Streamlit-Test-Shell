#!/usr/bin/env python3
"""
Unified Mapping Tool - Combines route mapping, intel injection, reach generation, and interactive viewing.

Usage:
    # Generate reach data from JSON
    python mapping_tool.py reach --json lstx-reasonable.json --output RedReachScenario1.csv
    
    # Create route map from CSV
    python mapping_tool.py route --csv routesTime001.csv --output route_map.html
    
    # Add intel nodes to existing map
    python mapping_tool.py intel --html route_map.html --csv Intel1.csv --output route_map_with_intel.html
    
    # Run interactive Streamlit viewer
    python mapping_tool.py viewer
    # or
    streamlit run mapping_tool.py viewer
"""

import pandas as pd
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import requests
import time
import sys
import math
import os
import base64
import re
import json
import numpy as np
import argparse
from datetime import datetime, timezone, timedelta

# OSRM API endpoint for routing (public instance, can be replaced with local)
OSRM_URL = "http://router.project-osrm.org/route/v1/driving"


# ============================================================================
# Utility Functions
# ============================================================================


def get_symbol_base64(symbol_name, base_dir):
    """Get base64 encoded image data for a symbol from Symbols directory.
    Looks in Symbols directory relative to script location first, then base_dir.
    """
    symbol_file = f"{symbol_name}.png"
    
    # First check Symbols directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(script_dir, "Symbols")
    symbol_path = os.path.join(symbols_dir, symbol_file)
    
    # If not found in Symbols, check base_dir
    if not os.path.exists(symbol_path) and base_dir:
        symbol_path = os.path.join(base_dir, "Symbols", symbol_file)
        # Also try directly in base_dir (for backward compatibility)
        if not os.path.exists(symbol_path):
            symbol_path = os.path.join(base_dir, symbol_file)
    
    if os.path.exists(symbol_path):
        try:
            with open(symbol_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            print(f"  Warning: Error loading symbol {symbol_path}: {e}")
    
    return None


def get_route(start_coords, end_coords):
    """Get route coordinates between two points using OSRM."""
    try:
        url = f"{OSRM_URL}/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
        params = {"overview": "full", "geometries": "geojson"}
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok" and len(data.get("routes", [])) > 0:
                route = data["routes"][0]
                geometry = route["geometry"]
                coordinates = [[coord[1], coord[0]] for coord in geometry["coordinates"]]
                return coordinates
    except Exception as e:
        print(f"Error getting route: {e}")
    
    return [start_coords, end_coords]


def create_curved_arc(start_coords, end_coords, num_points=60, curvature=0.3):
    """Create a curved arc between two points for abstract visualization."""
    mid_lat = (start_coords[0] + end_coords[0]) / 2
    mid_lon = (start_coords[1] + end_coords[1]) / 2
    
    dx = end_coords[1] - start_coords[1]
    dy = end_coords[0] - start_coords[0]
    distance = math.sqrt(dx**2 + dy**2)
    
    perp_dx = -dy / distance if distance > 0 else 0
    perp_dy = dx / distance if distance > 0 else 0
    
    curve_offset = distance * curvature
    control_lat = mid_lat + perp_dx * curve_offset
    control_lon = mid_lon + perp_dy * curve_offset
    
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        lat = (1-t)**2 * start_coords[0] + 2*(1-t)*t * control_lat + t**2 * end_coords[0]
        lon = (1-t)**2 * start_coords[1] + 2*(1-t)*t * control_lon + t**2 * end_coords[1]
        points.append([lat, lon])
    
    return points


def get_bearing(point1, point2):
    """Calculate bearing (direction) between two points in degrees."""
    lat1 = math.radians(point1[0])
    lat2 = math.radians(point2[0])
    dlon = math.radians(point2[1] - point1[1])
    
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing


def normalize_weight(weight, min_weight, max_weight):
    """Normalize weight to a line width between 3 and 20 pixels."""
    if max_weight == min_weight:
        return 10
    
    min_width = 5
    max_width = 15
    normalized = min_width + (weight - min_weight) * (max_width - min_width) / (max_weight - min_weight)
    return normalized


def get_reach_color(reach_value):
    """Get color based on reach value: 0=blue, 0.5=yellow, 1=red.
    
    Args:
        reach_value: A value between 0 and 1
    
    Returns:
        Hex color string (e.g., '#FF0000')
    """
    # Clamp reach_value to [0, 1]
    reach_value = max(0.0, min(1.0, reach_value))
    
    if reach_value <= 0.5:
        # Interpolate from blue (0,0,255) to yellow (255,255,0)
        ratio = reach_value / 0.5
        r = int(0 + 255 * ratio)
        g = int(0 + 255 * ratio)
        b = int(255 * (1 - ratio))
    else:
        # Interpolate from yellow (255,255,0) to red (255,0,0)
        ratio = (reach_value - 0.5) / 0.5
        r = 255
        g = int(255 * (1 - ratio))
        b = 0
    
    return f"#{r:02x}{g:02x}{b:02x}"


def get_weight_color(weight, min_weight, max_weight, use_blue=False):
    """Get color based on weight: red (max) -> orange -> yellow (min) or blue."""
    if use_blue:
        return '#0066FF'
    
    if max_weight == min_weight:
        return "#FF0000"
    
    normalized = (weight - min_weight) / (max_weight - min_weight)
    
    if normalized >= 0.5:
        ratio = (normalized - 0.5) / 0.5
        r = 255
        g = int(165 * (1 - ratio))
        b = 0
    else:
        ratio = normalized / 0.5
        r = 255
        g = int(165 + 90 * (1 - ratio))
        b = 0
    
    return f"#{r:02x}{g:02x}{b:02x}"


def get_location_icon(location_name, location_symbols, base_dir, is_origin=True):
    """Get icon for a location. Uses custom symbol if available."""
    symbol_key = location_symbols.get(location_name)
    
    if symbol_key:
        img_data_uri = get_symbol_base64(symbol_key, base_dir)
        if img_data_uri:
            return folium.CustomIcon(
                icon_image=img_data_uri,
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                popup_anchor=(0, -15)
            )
    
    default_color = 'green' if is_origin else 'red'
    return folium.Icon(color=default_color, icon='circle')


# ============================================================================
# Reach Data Generation
# ============================================================================

def generate_reach_dataframe(json_file, num_time_periods=39, rules=None):
    """Generate reach data from JSON file based on rules. Returns a DataFrame."""
    with open(json_file) as f:
        data = json.load(f)
    
    lPairs = data["location_pair_data"]
    
    rows = [
        {
            "Origin": p.get("origin_location_id"),
            "Destination": p.get("destination_location_id"),
        }
        for p in lPairs
    ]
    
    df = pd.DataFrame(rows)
    
    # Default rules for reach calculation
    if rules is None:
        rules = {
            "Medan_Air_Base": [
                {"start": 1, "end": 30, "type": "random", "low": 0.1, "high": 0.8},
                {"start": 31, "end": 44, "type": "const", "value": 0},
            ],
            "Phase_I__II_Recon_Deep_Staging_Location": [
                {"start": 1, "end": 30, "type": "random", "low": 0.1, "high": 0.8},
                {"start": 31, "end": 44, "type": "const", "value": 0},
            ],
            "Phase_II_D1_PAA_3": [
                {"start": 1, "end": 40, "type": "random", "low": 0.1, "high": 0.8},
                {"start": 41, "end": 44, "type": "const", "value": 0},
            ],
            "Phase_II_D2_PAA4": [
                {"start": 1, "end": 40, "type": "random", "low": 0.1, "high": 0.8},
                {"start": 41, "end": 44, "type": "const", "value": 0},
            ],
            "Phase_II_D3_PAA5": [
                {"start": 1, "end": 44, "type": "random", "low": 0.1, "high": 0.8},
            ],
            "Phase_II_D1_128__328_Supporting_Position": [
                {"start": 1, "end": 40, "type": "random", "low": 0.1, "high": 0.8},
                {"start": 41, "end": 44, "type": "const", "value": 0},
            ],
        }
    
    loc_masks = {loc: (df["Destination"] == loc) for loc in rules}
    
    for tp in range(1, num_time_periods + 1):
        col = np.zeros(len(df), dtype=float)
        
        for loc, segments in rules.items():
            mask_loc = loc_masks[loc]
            if not mask_loc.any():
                continue
            
            for seg in segments:
                if seg["start"] <= tp <= seg["end"]:
                    if seg["type"] == "const":
                        col[mask_loc] = seg["value"]
                    elif seg["type"] == "random":
                        low, high = seg["low"], seg["high"]
                        col[mask_loc] = low + np.random.rand(mask_loc.sum()) * (high - low)
                    break
        
        df[f"Reach TP{tp}"] = col
    
    return df


def convert_reach_dataframe_to_dict(df):
    """Convert reach DataFrame to dictionary format used by load_reach_data."""
    reach_dict = {}
    
    origin_col = None
    dest_col = None
    for col in df.columns:
        if col.lower() == 'origin':
            origin_col = col
        elif col.lower() == 'destination':
            dest_col = col
    
    if not origin_col or not dest_col:
        return None
    
    reach_cols = [col for col in df.columns if col.startswith('Reach TP')]
    
    for idx, row in df.iterrows():
        origin = str(row[origin_col]).strip()
        dest = str(row[dest_col]).strip()
        arc_key = (origin, dest)
        
        if arc_key not in reach_dict:
            reach_dict[arc_key] = {}
        for col in reach_cols:
            try:
                tp_num = int(col.replace('Reach TP', '').strip())
                reach_value = row[col]
                if pd.notna(reach_value):
                    reach_dict[arc_key][tp_num] = float(reach_value)
            except (ValueError, TypeError):
                continue
    
    return reach_dict


def generate_reach_data(json_file, output_file, num_time_periods=39):
    """Generate reach data from JSON file and save to CSV (CLI command)."""
    df = generate_reach_dataframe(json_file, num_time_periods)
    df.to_csv(output_file, index=False)
    print(f"Success: Reach data saved to {output_file}")


# ============================================================================
# Route Map Creation
# ============================================================================

def create_route_map(csv_file, output_file=None, use_blue_color=False, 
                     current_time=None, max_time=None, reach_data=None, time_horizon=None):
    """Create a folium map with routes from CSV data."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Require coordinates
    if not all(col in df.columns for col in ['originLat', 'originLong', 'destLat', 'destLong']):
        print("Error: CSV must contain originLat, originLong, destLat, destLong columns")
        return None
    
    location_col = 'l' if 'l' in df.columns else 'location'
    location_prime_col = 'lprime' if 'lprime' in df.columns else 'location_prime'
    weight_col = 'totalWeight' if 'totalWeight' in df.columns else 'weight_shipped'
    
    # Process locations and coordinates
    all_coords = {}
    location_symbols = {}
    has_symbols = 'sym1' in df.columns and 'sym2' in df.columns
    
    for idx, row in df.iterrows():
        origin_name = str(row[location_col])
        dest_name = str(row[location_prime_col])
        
        try:
            origin_coords = (float(row['originLat']), float(row['originLong']))
            dest_coords = (float(row['destLat']), float(row['destLong']))
        except (ValueError, TypeError):
            continue
        
        all_coords[origin_name] = origin_coords
        all_coords[dest_name] = dest_coords
        
        if has_symbols:
            sym1 = str(row['sym1']) if pd.notna(row['sym1']) else None
            sym2 = str(row['sym2']) if pd.notna(row['sym2']) else None
            if sym1:
                location_symbols[origin_name] = sym1
            if sym2:
                location_symbols[dest_name] = sym2
    
    # Calculate center
    if all_coords:
        center_lat = sum(c[0] for c in all_coords.values()) / len(all_coords)
        center_lon = sum(c[1] for c in all_coords.values()) / len(all_coords)
    else:
        center_lat, center_lon = 40.7128, -74.0060
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Collect all coordinates for bounds calculation
    all_points = []
    for coords in all_coords.values():
        all_points.append(coords)
    
    # Calculate min/max weights
    weights = []
    for idx, row in df.iterrows():
        start_loc = row[location_col]
        end_loc = row[location_prime_col]
        if start_loc in all_coords and end_loc in all_coords:
            try:
                weight = float(row[weight_col])
                weights.append(weight)
            except (ValueError, TypeError):
                pass
    
    if not weights:
        if output_file:
            m.save(output_file)
            print(f"Map saved to {output_file}")
        return m
    
    min_weight = min(weights)
    max_weight = max(weights)
    
    # Get base directory for symbols (use script directory where Symbols folder is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Track if any routes use reach coloring
    has_reach_coloring = False
    
    # Add routes
    route_index = 0
    for idx, row in df.iterrows():
        start_loc = row[location_col]
        end_loc = row[location_prime_col]
        
        if start_loc not in all_coords or end_loc not in all_coords:
            continue
        
        try:
            weight = float(row[weight_col])
        except (ValueError, TypeError):
            continue
        
        start_coords = all_coords[start_loc]
        end_coords = all_coords[end_loc]
        
        # Vary curvature
        base_curvature = 0.25
        curvature_variation = ((route_index % 7) - 3) * 0.08
        weight_factor = ((weight - min_weight) / (max_weight - min_weight)) * 0.1 if max_weight != min_weight else 0
        curvature = base_curvature + curvature_variation + weight_factor
        
        abstract_arc = create_curved_arc(start_coords, end_coords, num_points=50, curvature=curvature)
        all_points.extend(abstract_arc)
        
        line_width = normalize_weight(weight, min_weight, max_weight)
        
        # Build popup text with reach data if available
        popup_text = f"{start_loc} → {end_loc}<br>Weight: {weight:,.0f} lbs"
        tooltip_text = f"{start_loc} → {end_loc}"
        
        # Get reach value if available
        reach_value = None
        if reach_data is not None and current_time is not None:
            arc_key = (start_loc.strip(), end_loc.strip())
            if arc_key in reach_data:
                reach_value = reach_data[arc_key].get(current_time, None)
            else:
                for (orig, dest), time_data in reach_data.items():
                    if orig.strip().lower() == start_loc.strip().lower() and dest.strip().lower() == end_loc.strip().lower():
                        reach_value = time_data.get(current_time, None)
                        break
            
            if reach_value is not None:
                popup_text += f"<br>Reach: {reach_value:.4f}"
                tooltip_text += f"<br>Reach: {reach_value:.4f}"
        
        # Use reach value for coloring if available, otherwise use weight
        if reach_value is not None:
            line_color = get_reach_color(reach_value)
            has_reach_coloring = True
        else:
            line_color = get_weight_color(weight, min_weight, max_weight, use_blue_color)
        
        # Add arc with outline
        folium.PolyLine(
            abstract_arc,
            color='white',
            weight=line_width + 3,
            opacity=0.6,
            popup=popup_text,
            tooltip=tooltip_text
        ).add_to(m)
        
        folium.PolyLine(
            abstract_arc,
            color=line_color,
            weight=line_width,
            opacity=0.9,
            popup=popup_text,
            tooltip=tooltip_text
        ).add_to(m)
        
        route_index += 1
        
        # DEPRECATED: Directional arrows removed - they looked bad
        # Add markers
        origin_icon = get_location_icon(start_loc, location_symbols, base_dir, is_origin=True)
        dest_icon = get_location_icon(end_loc, location_symbols, base_dir, is_origin=False)
        
        folium.Marker(
            start_coords,
            popup=f"Origin: {start_loc}",
            icon=origin_icon
        ).add_to(m)
        
        folium.Marker(
            end_coords,
            popup=f"Destination: {end_loc}",
            icon=dest_icon
        ).add_to(m)
    
    # Auto-fit bounds
    if all_points:
        min_lat = min(p[0] for p in all_points)
        max_lat = max(p[0] for p in all_points)
        min_lon = min(p[1] for p in all_points)
        max_lon = max(p[1] for p in all_points)
        
        lat_padding = (max_lat - min_lat) * 0.1
        lon_padding = (max_lon - min_lon) * 0.1
        
        bounds = [
            [min_lat - lat_padding, min_lon - lon_padding],
            [max_lat + lat_padding, max_lon + lon_padding]
        ]
        m.fit_bounds(bounds)
    
    # Add north arrow and time overlay if time information is provided
    if current_time is not None and max_time is not None:
        time_text = None
        if time_horizon:
            pair = compute_time_display(
                time_horizon.get("log_start"),
                time_horizon.get("log_end"),
                current_time,
                max_time,
            )
            if pair:
                time_text = pair[0] + "<br>" + pair[1]
        add_north_arrow_and_time_overlay(m, current_time, max_time, time_text=time_text)
    
    # Add reach color legend if reach coloring is being used
    if has_reach_coloring:
        add_reach_color_legend(m)
    
    # Only save if output_file is provided (not None and not empty)
    if output_file:
        m.save(output_file)
        print(f"Map saved to {output_file}")
    
    return m


def add_north_arrow_and_time_overlay(m, current_time, max_time, time_text=None):
    """Add a north arrow with 'N' and time text in a white box.
    If time_text is provided (from time_horizon), shows rough actual time; else 'Time 001 / 039'.
    """
    from folium import MacroElement
    from jinja2 import Template

    if time_text is None:
        time_text = f"Time {current_time:03d} / {max_time:03d}"
    # JSON-encode so it can be embedded safely in JS as a string literal
    time_display_js = json.dumps(time_text)

    class NorthArrowOverlay(MacroElement):
        _template = Template("""
        {% macro script(this, kwargs) %}
            (function() {
                var northArrowDiv = document.createElement('div');
                northArrowDiv.id = 'north-arrow-overlay';
                northArrowDiv.style.cssText = 'position: absolute; bottom: 20px; left: 20px; background-color: white; border: 2px solid #333; border-radius: 5px; padding: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.3); z-index: 1000; font-family: Arial, sans-serif;';
                northArrowDiv.innerHTML = '<div style="text-align: center; margin-bottom: 5px;"><div style="font-size: 24px; font-weight: bold; color: #333; line-height: 1;">↑</div><div style="font-size: 14px; font-weight: bold; color: #333; margin-top: 2px;">N</div></div><div style="text-align: center; font-size: 12px; color: #333; border-top: 1px solid #ccc; padding-top: 5px; margin-top: 5px;">' + ({{ this.time_display_js | safe }}) + '</div>';
                var mapContainer = document.querySelector('.folium-map');
                if (mapContainer) {
                    mapContainer.style.position = 'relative';
                    mapContainer.appendChild(northArrowDiv);
                } else {
                    document.body.appendChild(northArrowDiv);
                }
                
                // Disable pointer events on all intel circles to prevent blocking arc hover
                setTimeout(function() {
                    var circles = document.querySelectorAll('svg path[fill="red"]');
                    circles.forEach(function(circle) {
                        var parent = circle.closest('svg');
                        if (parent && parent.getAttribute('class') && parent.getAttribute('class').includes('leaflet-interactive')) {
                            parent.style.pointerEvents = 'none';
                        }
                    });
                    // Also target circle elements directly
                    var circlePaths = document.querySelectorAll('svg.leaflet-interactive path');
                    circlePaths.forEach(function(path) {
                        var fill = path.getAttribute('fill');
                        if (fill === 'red' || fill === '#ff0000') {
                            path.style.pointerEvents = 'none';
                            var svg = path.closest('svg');
                            if (svg) {
                                svg.style.pointerEvents = 'none';
                            }
                        }
                    });
                }, 100);
            })();
        {% endmacro %}
        """)
        
        def __init__(self, time_display_js):
            super().__init__()
            self.time_display_js = time_display_js

    overlay = NorthArrowOverlay(time_display_js)
    overlay.add_to(m)


def add_reach_color_legend(m):
    """Add a color legend showing reach value to color mapping (blue=0, red=1)."""
    from folium import MacroElement
    from jinja2 import Template
    
    class ReachColorLegend(MacroElement):
        _template = Template("""
        {% macro script(this, kwargs) %}
            (function() {
                var legendDiv = document.createElement('div');
                legendDiv.id = 'reach-color-legend';
                legendDiv.style.cssText = 'position: absolute; top: 20px; right: 20px; background-color: white; border: 2px solid #333; border-radius: 5px; padding: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.3); z-index: 1000; font-family: Arial, sans-serif; min-width: 90px;';
                
                // Create gradient using CSS linear-gradient matching get_reach_color function
                var gradientHtml = '<div style="margin-bottom: 4px; font-weight: bold; font-size: 11px; color: #333; text-align: center;">Reach Value</div>';
                gradientHtml += '<div style="height: 120px; width: 20px; margin: 0 auto 4px auto; border: 1px solid #333; background: linear-gradient(to bottom, #0000FF 0%, #4040FF 12.5%, #8080FF 25%, #C0C0FF 37.5%, #FFFF00 50%, #FFC000 62.5%, #FF8000 75%, #FF4000 87.5%, #FF0000 100%);"></div>';
                
                // Labels
                gradientHtml += '<div style="display: flex; justify-content: space-between; font-size: 9px; color: #333; margin-top: 2px;">';
                gradientHtml += '<span>0.0</span><span>0.5</span><span>1.0</span>';
                gradientHtml += '</div>';
                
                // Color description
                gradientHtml += '<div style="font-size: 8px; color: #666; text-align: center; margin-top: 4px; border-top: 1px solid #ccc; padding-top: 4px;">Blue=0 Yel=0.5 Red=1</div>';
                
                legendDiv.innerHTML = gradientHtml;
                var mapContainer = document.querySelector('.folium-map');
                if (mapContainer) {
                    mapContainer.style.position = 'relative';
                    mapContainer.appendChild(legendDiv);
                } else {
                    document.body.appendChild(legendDiv);
                }
            })();
        {% endmacro %}
        """)
    
    legend = ReachColorLegend()
    legend.add_to(m)


def add_circle_pointer_events_disabler(m):
    """Add JavaScript to disable pointer events on intel circles."""
    from folium import MacroElement
    from jinja2 import Template
    
    class CircleDisabler(MacroElement):
        _template = Template("""
        {% macro script(this, kwargs) %}
            (function() {
                function disableCircleEvents() {
                    // Find all circle SVG elements and disable pointer events
                    var allPaths = document.querySelectorAll('svg.leaflet-interactive path');
                    allPaths.forEach(function(path) {
                        var fill = path.getAttribute('fill');
                        var fillOpacity = path.getAttribute('fill-opacity') || path.getAttribute('fillOpacity');
                        // Check if it's a red circle (intel radius circle)
                        if ((fill === 'red' || fill === '#ff0000') && fillOpacity && parseFloat(fillOpacity) < 1) {
                            path.style.pointerEvents = 'none';
                            var svg = path.closest('svg.leaflet-interactive');
                            if (svg) {
                                svg.style.pointerEvents = 'none';
                            }
                        }
                    });
                }
                
                // Run immediately and after a delay to catch dynamically added circles
                disableCircleEvents();
                setTimeout(disableCircleEvents, 100);
                setTimeout(disableCircleEvents, 500);
            })();
        {% endmacro %}
        """)
    
    disabler = CircleDisabler()
    disabler.add_to(m)


# ============================================================================
# Intel Injection
# ============================================================================

def add_intel_nodes_to_map(m, intel_df, base_dir, csv_suffix=""):
    """Add intelligence nodes with radius circles to an existing folium map."""
    symbol_col = None
    radius_col = None
    lat_col = None
    lon_col = None
    
    for col in intel_df.columns:
        col_lower = col.lower()
        if col_lower in ['symbol', 'sym1', 'sym'] or 'symbol' in col_lower:
            symbol_col = col
        if 'radius' in col_lower or 'engagement' in col_lower:
            radius_col = col
        if col_lower in ['lat', 'latitude']:
            lat_col = col
        if col_lower in ['long', 'lon', 'longitude']:
            lon_col = col
    
    if not radius_col or not lat_col or not lon_col:
        return m, []
    
    base_timestamp = int(time.time() * 1000000)
    intel_coords = []
    
    for idx, row in intel_df.iterrows():
        location_name = str(row['location']) if 'location' in intel_df.columns else f"Intel {idx}"
        symbol_name = str(row[symbol_col]) if symbol_col and pd.notna(row[symbol_col]) else None
        radius_km = float(row[radius_col]) if pd.notna(row[radius_col]) else 0
        
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            coords = (lat, lon)
        except (ValueError, TypeError):
            continue
        
        intel_coords.append(coords)
        
        if radius_km > 0:
            radius_m = radius_km * 1000
            for angle in range(0, 360, 45):
                lat_offset = (radius_m / 111320) * math.cos(math.radians(angle))
                lon_offset = (radius_m / (111320 * math.cos(math.radians(coords[0])))) * math.sin(math.radians(angle))
                intel_coords.append((coords[0] + lat_offset, coords[1] + lon_offset))
        
        unique_id = f"{csv_suffix}_{base_timestamp}_{idx}_{abs(hash(str(coords) + location_name))}"
        
        if symbol_name:
            img_data_uri = get_symbol_base64(symbol_name, base_dir)
            if img_data_uri:
                icon = folium.CustomIcon(
                    icon_image=img_data_uri,
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                    popup_anchor=(0, -15)
                )
            else:
                icon = folium.Icon(color='blue', icon='info-sign')
        else:
            icon = folium.Icon(color='blue', icon='info-sign')
        
        popup_text = f"Intel: {location_name}<br>Radius: {radius_km} km<br>ID: {unique_id}"
        marker = folium.Marker(
            location=coords,
            popup=folium.Popup(popup_text, parse_html=False),
            icon=icon,
            tooltip=f"Intel: {location_name}",
            name=f"intel_marker_{unique_id}"
        )
        marker.add_to(m)
        
        if radius_km > 0:
            radius_m = radius_km * 1000
            circle = folium.Circle(
                location=coords,
                radius=radius_m,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.2,
                weight=2,
                interactive=False,  # Make non-interactive so it doesn't block arc hover
                name=f"intel_circle_{unique_id}"
            )
            circle.add_to(m)
    
    # Add JavaScript to disable pointer events on circles
    add_circle_pointer_events_disabler(m)
    
    return m, intel_coords


def add_intel_nodes_to_html(html_file, csv_file, output_file=None):
    """Add intelligence nodes to an existing HTML map file."""
    if output_file is None:
        output_file = html_file
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    if 'location' not in df.columns:
        print("Error: CSV must contain 'location' column")
        sys.exit(1)
    
    symbol_col = None
    radius_col = None
    lat_col = None
    lon_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['symbol', 'sym1', 'sym'] or 'symbol' in col_lower:
            symbol_col = col
        if 'radius' in col_lower or 'engagement' in col_lower:
            radius_col = col
        if col_lower in ['lat', 'latitude']:
            lat_col = col
        if col_lower in ['long', 'lon', 'longitude']:
            lon_col = col
    
    if not radius_col:
        print("Error: No radius/engagement column found in CSV")
        sys.exit(1)
    
    has_coordinates = lat_col is not None and lon_col is not None
    
    csv_dir = os.path.dirname(os.path.abspath(csv_file))
    base_dir = os.path.dirname(os.path.abspath(__file__))
    geolocator = None
    if not has_coordinates:
        geolocator = Nominatim(user_agent="mapping_tool")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    map_var_match = re.search(r'var (map_\w+) = L\.map', html_content)
    if not map_var_match:
        print("Error: Could not find map variable in HTML file")
        sys.exit(1)
    
    map_var = map_var_match.group(1)
    
    if has_coordinates:
        print("Using coordinates from CSV...")
    else:
        print("Geocoding locations...")
    
    js_code = "\n        // Intel injector additions\n"
    
    for idx, row in df.iterrows():
        location_name = str(row['location'])
        symbol_name = str(row[symbol_col]) if symbol_col and pd.notna(row[symbol_col]) else None
        radius_km = float(row[radius_col]) if pd.notna(row[radius_col]) else 0
        
        print(f"  Processing {location_name}...")
        
        if has_coordinates:
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                coords = (lat, lon)
            except (ValueError, TypeError):
                print(f"    Warning: Invalid coordinates for '{location_name}', skipping")
                continue
        else:
            coords = geocode_location(location_name + ", New York, USA", geolocator)
            if not coords:
                print(f"    Warning: Could not geocode '{location_name}'")
                continue
            lat, lon = coords
        
        location_name_escaped = location_name.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
        
        marker_var = f"intel_marker_{abs(hash(location_name))}"
        if symbol_name:
            img_data_uri = get_symbol_base64(symbol_name, base_dir)
            if img_data_uri:
                icon_var = f"intel_icon_{abs(hash(location_name))}"
                img_data_uri_escaped = img_data_uri.replace("'", "\\'")
                js_code += f"""
        var {icon_var} = L.icon({{
            iconUrl: '{img_data_uri_escaped}',
            iconSize: [30, 30],
            iconAnchor: [15, 15],
            popupAnchor: [0, -15]
        }});
        var {marker_var} = L.marker([{lat}, {lon}], {{icon: {icon_var}}}).addTo({map_var});
        {marker_var}.bindPopup('Intel: {location_name_escaped}<br>Radius: {radius_km} km');
"""
            else:
                js_code += f"""
        var {marker_var} = L.marker([{lat}, {lon}]).addTo({map_var});
        {marker_var}.bindPopup('Intel: {location_name_escaped}<br>Radius: {radius_km} km');
"""
        else:
            js_code += f"""
        var {marker_var} = L.marker([{lat}, {lon}]).addTo({map_var});
        {marker_var}.bindPopup('Intel: {location_name_escaped}<br>Radius: {radius_km} km');
"""
        
        if radius_km > 0:
            radius_m = radius_km * 1000
            circle_var = f"intel_circle_{abs(hash(location_name))}"
            js_code += f"""
        var {circle_var} = L.circle([{lat}, {lon}], {{
            color: 'red',
            fillColor: 'red',
            fillOpacity: 0.2,
            radius: {radius_m},
            weight: 2,
            interactive: false,
            bubblingMouseEvents: false
        }}).addTo({map_var});
        {circle_var}.off('mouseover mouseout click');
        setTimeout(function() {{
            var svg = {circle_var}.getElement();
            if (svg) {{
                svg.style.pointerEvents = 'none';
                var paths = svg.querySelectorAll('path');
                paths.forEach(function(p) {{ p.style.pointerEvents = 'none'; }});
            }}
        }}, 100);
"""
        
        time.sleep(0.5)
    
    script_matches = list(re.finditer(r'</script>', html_content))
    
    if script_matches:
        last_script_match = script_matches[-1]
        insert_pos = last_script_match.start()
        html_content = html_content[:insert_pos] + js_code + '\n        ' + html_content[insert_pos:]
    else:
        html_content = html_content.replace('</html>', js_code + '</html>')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nIntel nodes added to {output_file}")


# ============================================================================
# MIP Output Processing
# ============================================================================

def generate_symbol_mapping_from_json(json_file):
    """Generate symbol mapping from JSON file location data (incorporates symbol-injector logic).
    
    Returns DataFrame with location and Symbol columns.
    """
    with open(json_file) as f:
        data = json.load(f)
    
    # Extract location IDs and assign default symbol (like symbol-injector.py)
    locations = [loc["id"] for loc in data["locations"]]
    symbols = ["SupplyCompany"] * len(locations)  # Default symbol as per symbol-injector.py
    
    df_symbols = pd.DataFrame({
        "location": locations,
        "Symbol": symbols
    })
    
    return df_symbols


def process_mip_output_for_time(mip_file, json_file, time_period, symbol_file=None):
    """Process MIP output file with JSON scenario to create route dataframe for a specific time period.
    
    Args:
        mip_file: Path to MIP output CSV (e.g., M0.csv, M3.csv)
        json_file: Path to JSON scenario file
        time_period: Time period number to extract
        symbol_file: Optional path to symbol CSV file
    
    Returns:
        DataFrame with columns: l, originLat, originLong, lprime, destLat, destLong, t, totalWeight, sym1, sym2
    """
    # Load location data from JSON
    with open(json_file) as f:
        data = json.load(f)
    
    df_locations = pd.DataFrame(data["locations"])[["id", "latitude", "longitude"]]
    df_locations["id"] = df_locations["id"].astype(str).str.strip()
    
    # Create lookup tables
    origin_lookup = df_locations.rename(columns={
        "id": "l",
        "latitude": "originLat",
        "longitude": "originLong"
    })
    
    dest_lookup = df_locations.rename(columns={
        "id": "lprime",
        "latitude": "destLat",
        "longitude": "destLong"
    })
    
    # Load MIP output
    df_mip = pd.read_csv(mip_file)
    df_mip.columns = df_mip.columns.str.strip()
    
    # Calculate weight per row
    df_mip["WeightT"] = df_mip[["Mvalue", "per-item weight"]].prod(axis=1)
    
    # Filter for specific time period
    df_t = df_mip[df_mip["t"] == time_period].copy()
    
    if df_t.empty:
        return pd.DataFrame(columns=["l", "originLat", "originLong", "lprime", "destLat", "destLong", 
                                     "t", "totalWeight", "sym1", "sym2"])
    
    # Aggregate by route
    df_agg = (
        df_t.groupby(["l", "lprime", "t"], as_index=False)
            .agg(totalWeight=("WeightT", "sum"))
    )
    
    # Normalize route ids
    df_agg["l"] = df_agg["l"].astype(str).str.strip()
    df_agg["lprime"] = df_agg["lprime"].astype(str).str.strip()
    
    # Merge with location lookups
    df_out = df_agg.merge(origin_lookup, on="l", how="left")
    df_out = df_out.merge(dest_lookup, on="lprime", how="left")
    
    # Load or generate symbol mapping
    df_symbols = None
    if symbol_file and os.path.exists(symbol_file):
        try:
            df_symbols = pd.read_csv(symbol_file)
            df_symbols.columns = df_symbols.columns.str.strip()
            df_symbols["location"] = df_symbols["location"].astype(str).str.strip()
        except Exception as e:
            print(f"Warning: Could not load symbol file {symbol_file}: {e}")
    
    # Generate from JSON if no symbol file
    if df_symbols is None:
        try:
            df_symbols = generate_symbol_mapping_from_json(json_file)
        except Exception as e:
            print(f"Warning: Could not generate symbol mapping from JSON: {e}")
    
    # Merge symbols if available
    if df_symbols is not None and "Symbol" in df_symbols.columns:
        df_symbols["location"] = df_symbols["location"].astype(str).str.strip()
        df_symbols["Symbol"] = df_symbols["Symbol"].astype(str).str.strip().replace(["", "nan", "None"], None)
        
        origin_lookup = df_symbols.rename(columns={"location": "l", "Symbol": "sym1"})[["l", "sym1"]]
        dest_lookup = df_symbols.rename(columns={"location": "lprime", "Symbol": "sym2"})[["lprime", "sym2"]]
        
        df_out = df_out.merge(origin_lookup, on="l", how="left").merge(dest_lookup, on="lprime", how="left")
    else:
        df_out["sym1"] = df_out["sym2"] = None
    
    # Reorder columns
    df_out = df_out[[
        "l", "originLat", "originLong",
        "lprime", "destLat", "destLong",
        "t", "totalWeight",
        "sym1", "sym2"
    ]]
    
    return df_out


# ============================================================================
# Streamlit Viewer
# ============================================================================

def load_time_horizon(json_path):
    """Load time_horizon (log_start, log_end) from a scenario JSON. Returns dict or None."""
    if not json_path or not os.path.exists(json_path):
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
        th = data.get("time_horizon")
        if th and "log_start" in th and "log_end" in th:
            return th
    except Exception:
        pass
    return None


def compute_time_display(log_start, log_end, current_time, max_time):
    """
    Compute rough actual time strings for the compass overlay from time_horizon.
    Period 1 = log_start, period max_time = log_end; each period is ~(duration/(max_time-1)).
    Returns (current_str, range_str) or None on parse error.
    """
    try:
        # Parse ISO strings; treat Z as UTC
        def parse(s):
            s = (s or "").strip().replace("Z", "+00:00")
            return datetime.fromisoformat(s)
        start = parse(log_start)
        end = parse(log_end)
        if start >= end or max_time < 1:
            return None
        total_seconds = (end - start).total_seconds()
        num_intervals = max(1, max_time - 1)
        seconds_per_period = total_seconds / num_intervals
        # Period N (1-based) is at start + (N-1) * seconds_per_period
        display_dt = start + timedelta(seconds=(current_time - 1) * seconds_per_period)
        # Clip to [start, end]
        if display_dt < start:
            display_dt = start
        if display_dt > end:
            display_dt = end
        current_str = "~" + display_dt.strftime("%b %d, %H:%M")
        total_days = total_seconds / (24.0 * 3600)
        hours_per_period = (total_seconds / 3600.0) / num_intervals
        range_str = start.strftime("%b %d") + " – " + end.strftime("%b %d")
        range_str += f" ({total_days:.0f}d, ~{hours_per_period:.1f}h/period)"
        return (current_str, range_str)
    except Exception:
        return None


def load_reach_data(reach_csv_path):
    """Load reach data from CSV file."""
    if not os.path.exists(reach_csv_path):
        return None
    
    try:
        df = pd.read_csv(reach_csv_path)
        reach_dict = {}
        
        origin_col = None
        dest_col = None
        for col in df.columns:
            if col.lower() == 'origin':
                origin_col = col
            elif col.lower() == 'destination':
                dest_col = col
        
        if not origin_col or not dest_col:
            return None
        
        reach_cols = [col for col in df.columns if col.startswith('Reach TP')]
        
        for idx, row in df.iterrows():
            origin = str(row[origin_col]).strip()
            dest = str(row[dest_col]).strip()
            arc_key = (origin, dest)
            
            if arc_key not in reach_dict:
                reach_dict[arc_key] = {}
            for col in reach_cols:
                try:
                    tp_num = int(col.replace('Reach TP', '').strip())
                    reach_value = row[col]
                    if pd.notna(reach_value):
                        reach_dict[arc_key][tp_num] = float(reach_value)
                except (ValueError, TypeError):
                    continue
        
        return reach_dict
    except Exception as e:
        print(f"Error loading reach data: {str(e)}")
        return None


def get_files_from_directory(directory, extension='.csv'):
    """Get list of files with given extension from directory."""
    if not os.path.exists(directory):
        return []
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    return sorted(files)




def run_streamlit_viewer():
    """Run the Streamlit interactive viewer."""
    try:
        import streamlit as st
        from streamlit_folium import st_folium
    except ImportError:
        print("Error: streamlit and streamlit-folium are required for the viewer mode.")
        print("Install with: pip install streamlit streamlit-folium")
        sys.exit(1)
    
    st.set_page_config(
        page_title="Route & Intel Map Viewer",
        layout="wide"
    )
   
   
   # Base directories loading 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mip_outputs_dir = os.path.join(base_dir, "MIPOutputs")
    json_scenarios_dir = os.path.join(base_dir, "JSON Scenarios")
    intel_dir = os.path.join(base_dir, "Intel_injects")
    reach_scenarios_dir = os.path.join(base_dir, "ReachScenarios")
    temp_html_dir = os.path.join(base_dir, "savedHTML")
    os.makedirs(mip_outputs_dir, exist_ok=True)
    os.makedirs(json_scenarios_dir, exist_ok=True)
    os.makedirs(intel_dir, exist_ok=True)
    os.makedirs(reach_scenarios_dir, exist_ok=True)
    os.makedirs(temp_html_dir, exist_ok=True)
    # Get available MIP files, JSON scenarios, and reach scenarios
    mip_files = get_files_from_directory(mip_outputs_dir, '.csv')
    json_files = get_files_from_directory(json_scenarios_dir, '.json')
    intel_csv_files = get_files_from_directory(intel_dir, '.csv')
    reach_csv_files = get_files_from_directory(reach_scenarios_dir, '.csv')
    
    # Filter MIP files to only M0 and M3
    # available_mips = [f for f in mip_files if f in ['M0.csv', 'M3.csv']]
    available_mips = mip_files # deprecated but kept for now
    
    # Selectors - 5 columns now
    selector_col1, selector_col2, selector_col3, selector_col4, selector_col5 = st.columns([1, 1, 1, 1, 1])
    
    with selector_col1:
        st.subheader("MIP Output")
        if not available_mips:
            st.warning("No MIP Outputs found.")
            selected_mip = None
        else:
            selected_mip = st.selectbox(
                "Run Selection:",
                options=available_mips,
                key="mip_select",
                label_visibility="collapsed"
            )
    
    with selector_col2:
        st.subheader("Scenario")
        if not json_files:
            st.warning("No JSON scenarios found.")
            selected_json = None
        else:
            selected_json = st.selectbox(
                "Select JSON:",
                options=json_files,
                key="json_select",
                label_visibility="collapsed"
            )
    '''
    This needs to be fixed it is not properly finding the max time from the MIP file and only using the default 39
    '''
    with selector_col3:
        st.subheader("Time Period")
        # Determine max time from MIP file if available
        max_time = 39  # default
        if selected_mip:
            mip_path = os.path.join(mip_outputs_dir, selected_mip)
            try:
                df_mip = pd.read_csv(mip_path)
                if 't' in df_mip.columns:
                    max_time = int(df_mip['t'].max()) 
            except:
                pass
        
        selected_time = st.slider(
            "Time:",
            min_value=1,
            max_value=max_time,
            value=1,
            key="time_slider",
            label_visibility="collapsed"
        )
        
        # Compute hours per period from scenario time_horizon when available
        hours_per_period_caption = "One time period = — hours"
        if selected_json:
            json_path = os.path.join(json_scenarios_dir, selected_json)
            th = load_time_horizon(json_path)
            if th and "log_start" in th and "log_end" in th:
                try:
                    def _parse(s):
                        s = (s or "").strip().replace("Z", "+00:00")
                        return datetime.fromisoformat(s)
                    start = _parse(th["log_start"])
                    end = _parse(th["log_end"])
                    if start < end and max_time >= 1:
                        total_seconds = (end - start).total_seconds()
                        num_intervals = max(1, max_time - 1)
                        hours_per_period = (total_seconds / 3600.0) / num_intervals
                        hours_per_period_caption = f"One time period = {hours_per_period:.1f} hours"
                except Exception:
                    pass
        st.caption(hours_per_period_caption)
    
    def get_optional_selectbox(files, key_prefix, subheader):
        """Helper to create optional selectbox with None option."""
        if not files:
            st.warning(f"No {subheader.lower()} files found.")
            return None
        selected = st.selectbox(
            f"Select {subheader}:",
            options=["None"] + files,
            key=f"{key_prefix}_select",
            label_visibility="collapsed"
        )
        return None if selected == "None" else selected
    
    with selector_col4:
        st.subheader("Intel Inject")
        selected_intel_csv = get_optional_selectbox(intel_csv_files, "intel_csv", "Intel CSV")
        st.caption("Optional")
    
    with selector_col5:
        st.subheader("Red Reach")
        selected_reach_csv = get_optional_selectbox(reach_csv_files, "reach_csv", "Reach CSV")
        st.caption("Optional")
    
    # Load reach data from selected CSV
    reach_data = None
    if selected_reach_csv:
        reach_csv_path = os.path.join(reach_scenarios_dir, selected_reach_csv)
        if 'reach_data_cache' not in st.session_state:
            st.session_state.reach_data_cache = {}
        
        if selected_reach_csv not in st.session_state.reach_data_cache:
            if os.path.exists(reach_csv_path):
                with st.spinner("Loading reach scenario..."):
                    try:
                        reach_data = load_reach_data(reach_csv_path)
                        if reach_data:
                            st.session_state.reach_data_cache[selected_reach_csv] = reach_data
                        else:
                            st.warning(f"Could not load reach data from {selected_reach_csv}")
                    except Exception as e:
                        st.error(f"Error loading reach scenario: {str(e)}")
            else:
                st.error(f"Reach file not found: {selected_reach_csv}")
        else:
            reach_data = st.session_state.reach_data_cache[selected_reach_csv]
    
    st.markdown("---")
    
    # Generate and display map
    if selected_mip and selected_json:
        mip_path = os.path.join(mip_outputs_dir, selected_mip)
        json_path = os.path.join(json_scenarios_dir, selected_json)
        
        if not os.path.exists(mip_path):
            st.error(f"MIP file not found: {selected_mip}")
        elif not os.path.exists(json_path):
            st.error(f"JSON file not found: {selected_json}")
        else:
            try:
                # Create unique key for caching
                map_key = f"{selected_mip}_{selected_json}_{selected_time}_{selected_intel_csv or 'none'}_{selected_reach_csv or 'none'}"
                
                if 'current_map_key' not in st.session_state or st.session_state.current_map_key != map_key:
                    with st.spinner("Processing MIP output and generating map..."):
                        # Look for symbol file
                        symbol_file = next(
                            (sf for sf in [
                                os.path.join(base_dir, "symbol-injector-output.csv"),
                                os.path.join(base_dir, "symbols.csv"),
                                os.path.join(json_scenarios_dir, "symbol-injector-output.csv"),
                            ] if os.path.exists(sf)),
                            None
                        )
                        
                        # Process MIP output for selected time period
                        route_df = process_mip_output_for_time(mip_path, json_path, selected_time, symbol_file=symbol_file)
                        
                        if route_df.empty:
                            st.warning(f"No routes found for time period {selected_time}")
                            base_map = None
                        else:
                            # Save to temporary CSV for create_route_map
                            import tempfile
                            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                            route_df.to_csv(temp_csv.name, index=False)
                            temp_csv.close()
                            
                            # Determine color scheme (M0 = blue, M3 = red/orange/yellow gradient)
                            use_blue = selected_mip == "M0.csv"
                            
                            # Load time_horizon from JSON for compass time display
                            time_horizon = load_time_horizon(json_path)
                            
                            # Create base map
                            base_map = create_route_map(temp_csv.name, use_blue_color=use_blue,
                                                       current_time=selected_time, max_time=max_time,
                                                       reach_data=reach_data, time_horizon=time_horizon)
                            
                            # Add intel nodes if selected
                            all_intel_coords = []
                            if base_map and selected_intel_csv:
                                intel_csv_path = os.path.join(intel_dir, selected_intel_csv)
                                if os.path.exists(intel_csv_path):
                                    try:
                                        intel_df = pd.read_csv(intel_csv_path)
                                        if not intel_df.empty:
                                            import time as time_module
                                            csv_suffix = f"{os.path.splitext(selected_mip)[0]}_{int(time_module.time() * 1000000)}_{os.path.splitext(selected_intel_csv)[0]}"
                                            base_map, intel_coords = add_intel_nodes_to_map(base_map, intel_df, base_dir, csv_suffix=csv_suffix)
                                            all_intel_coords.extend(intel_coords)
                                    except Exception as e:
                                        st.error(f"Error processing intel: {str(e)}")
                            
                            # Fit bounds to include all points
                            if base_map:
                                route_coords = [
                                    (float(row['originLat']), float(row['originLong']))
                                    for _, row in route_df.iterrows()
                                    if all(k in row for k in ['originLat', 'originLong'])
                                ] + [
                                    (float(row['destLat']), float(row['destLong']))
                                    for _, row in route_df.iterrows()
                                    if all(k in row for k in ['destLat', 'destLong'])
                                ]
                                
                                all_points = route_coords + all_intel_coords
                                if all_points:
                                    lats = [p[0] for p in all_points]
                                    lons = [p[1] for p in all_points]
                                    lat_range = max(lats) - min(lats)
                                    lon_range = max(lons) - min(lons)
                                    bounds = [
                                        [min(lats) - max(lat_range * 0.1, 0.01), min(lons) - max(lon_range * 0.1, 0.01)],
                                        [max(lats) + max(lat_range * 0.1, 0.01), max(lons) + max(lon_range * 0.1, 0.01)]
                                    ]
                                    base_map.fit_bounds(bounds)
                            
                            # Clean up temp file
                            try:
                                os.unlink(temp_csv.name)
                            except:
                                pass
                            
                            st.session_state.current_map_key = map_key
                            st.session_state.current_map = base_map
                else:
                    base_map = st.session_state.current_map
                
                if base_map:
                    st_folium(base_map, width=None, height=700, returned_objects=[])
                    
                    # Save buttons
                    save_col1, save_col2, save_col3 = st.columns([1, 1, 3])
                    map_filename = f"{os.path.splitext(selected_mip)[0]}_{os.path.splitext(selected_json)[0]}_t{selected_time:03d}"
                    if selected_intel_csv:
                        map_filename += f"_{os.path.splitext(selected_intel_csv)[0]}"
                    html_filename = map_filename + '.html'
                    html_path = os.path.join(temp_html_dir, html_filename)
                    image_filename = map_filename + '.png'
                    image_path = os.path.join(temp_html_dir, image_filename)
                    
                    with save_col1:
                        if st.button("Save Full Map", key="save_map"):
                            import tempfile as tf
                            symbol_file = next(
                                (sf for sf in [
                                    os.path.join(base_dir, "symbol-injector-output.csv"),
                                    os.path.join(base_dir, "symbols.csv"),
                                    os.path.join(json_scenarios_dir, "symbol-injector-output.csv"),
                                ] if os.path.exists(sf)),
                                None
                            )
                            time_horizon = load_time_horizon(json_path)
                            use_blue = selected_mip == "M0.csv"
                            intel_df = None
                            if selected_intel_csv:
                                intel_csv_path = os.path.join(intel_dir, selected_intel_csv)
                                if os.path.exists(intel_csv_path):
                                    try:
                                        intel_df = pd.read_csv(intel_csv_path)
                                    except Exception:
                                        intel_df = None
                            progress_bar = st.progress(0.0)
                            saved_count = 0
                            for t in range(1, max_time + 1):
                                route_df = process_mip_output_for_time(mip_path, json_path, t, symbol_file=symbol_file)
                                if route_df.empty:
                                    progress_bar.progress(t / max_time)
                                    continue
                                temp_csv = tf.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                                route_df.to_csv(temp_csv.name, index=False)
                                temp_csv.close()
                                try:
                                    m = create_route_map(temp_csv.name, use_blue_color=use_blue,
                                                        current_time=t, max_time=max_time,
                                                        reach_data=reach_data, time_horizon=time_horizon)
                                    all_intel_coords = []
                                    if m and intel_df is not None and not intel_df.empty:
                                        import time as time_module
                                        csv_suffix = f"{os.path.splitext(selected_mip)[0]}_t{t:03d}_{int(time_module.time() * 1000000)}_{os.path.splitext(selected_intel_csv)[0]}"
                                        m, intel_coords = add_intel_nodes_to_map(m, intel_df, base_dir, csv_suffix=csv_suffix)
                                        all_intel_coords = intel_coords
                                    route_coords = [
                                        (float(row['originLat']), float(row['originLong']))
                                        for _, row in route_df.iterrows()
                                        if all(k in row for k in ['originLat', 'originLong'])
                                    ] + [
                                        (float(row['destLat']), float(row['destLong']))
                                        for _, row in route_df.iterrows()
                                        if all(k in row for k in ['destLat', 'destLong'])
                                    ]
                                    all_pts = route_coords + all_intel_coords
                                    if all_pts and m:
                                        lats = [p[0] for p in all_pts]
                                        lons = [p[1] for p in all_pts]
                                        lat_r = max(lats) - min(lats)
                                        lon_r = max(lons) - min(lons)
                                        m.fit_bounds([
                                            [min(lats) - max(lat_r * 0.1, 0.01), min(lons) - max(lon_r * 0.1, 0.01)],
                                            [max(lats) + max(lat_r * 0.1, 0.01), max(lons) + max(lon_r * 0.1, 0.01)]
                                        ])
                                    if m:
                                        map_filename = f"{os.path.splitext(selected_mip)[0]}_{os.path.splitext(selected_json)[0]}_t{t:03d}"
                                        if selected_intel_csv:
                                            map_filename += f"_{os.path.splitext(selected_intel_csv)[0]}"
                                        map_filename += ".html"
                                        out_path = os.path.join(temp_html_dir, map_filename)
                                        m.save(out_path)
                                        saved_count += 1
                                finally:
                                    try:
                                        os.unlink(temp_csv.name)
                                    except Exception:
                                        pass
                                progress_bar.progress(t / max_time)
                            progress_bar.empty()
                            skipped = max_time - saved_count
                            if skipped > 0:
                                st.success(f"Saved {saved_count} maps to savedHTML/ ({skipped} periods had no routes).")
                            else:
                                st.success(f"Saved {saved_count} maps to savedHTML/!")
                            st.rerun()
                    
                    with save_col2:
                        if st.button("Save Image", key="save_image"):
                            import tempfile
                            temp_html = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
                            try:
                                base_map.save(temp_html.name)
                                temp_html.close()
                                
                                # Try html2image first, fallback to selenium
                                try:
                                    from html2image import Html2Image
                                    hti = Html2Image(size=(1920, 1080))
                                    hti.screenshot(html_file=temp_html.name, save_as=os.path.basename(image_path), size=(1920, 1080))
                                    import shutil
                                    temp_img = os.path.join(os.getcwd(), os.path.basename(image_path))
                                    if os.path.exists(temp_img):
                                        shutil.move(temp_img, image_path)
                                except ImportError:
                                    from selenium import webdriver
                                    from selenium.webdriver.chrome.options import Options
                                    import time as time_module
                                    
                                    chrome_options = Options()
                                    for arg in ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu', '--window-size=1920,1080']:
                                        chrome_options.add_argument(arg)
                                    
                                    try:
                                        from selenium.webdriver.chrome.service import Service
                                        from webdriver_manager.chrome import ChromeDriverManager
                                        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
                                    except Exception:
                                        driver = webdriver.Chrome(options=chrome_options)
                                    
                                    driver.get(f"file://{os.path.abspath(temp_html.name)}")
                                    time_module.sleep(3)
                                    driver.save_screenshot(image_path)
                                    driver.quit()
                                
                                st.success(f"Saved image to {image_filename}!")
                                st.rerun()
                            except ImportError:
                                st.error("html2image or selenium required. Install: pip install html2image (or pip install selenium webdriver-manager)")
                            except Exception as e:
                                st.error(f"Error saving image: {str(e)}")
                            finally:
                                try:
                                    os.unlink(temp_html.name)
                                except:
                                    pass
                else:
                    st.error("Failed to generate map")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Please select both MIP output (M0 or M3) and JSON scenario to display the map.")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    """Main function that works both as CLI and as a Streamlit page function."""
    # Check if we're being called from Streamlit (when called from g4t3_app.py)
    # When called as a function from Streamlit, we need to detect that context
    try:
        import streamlit as st
        # Check if any of our CLI commands are in sys.argv
        # If not, and streamlit is available, we're likely being called from Streamlit
        cli_commands = ['reach', 'route', 'intel', 'viewer']
        has_cli_command = any(cmd in sys.argv for cmd in cli_commands)
        
        # If streamlit is available and no CLI command is present, run viewer
        if not has_cli_command:
            run_streamlit_viewer()
            return
        # If 'viewer' is explicitly requested, run it
        if 'viewer' in sys.argv:
            run_streamlit_viewer()
            return
    except (ImportError, RuntimeError):
        # Not in Streamlit context, continue with CLI parsing below
        pass
    
    # CLI mode - parse arguments
    parser = argparse.ArgumentParser(description="Unified Mapping Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Reach command
    reach_parser = subparsers.add_parser('reach', help='Generate reach data from JSON')
    reach_parser.add_argument('--json', required=True, help='Input JSON file')
    reach_parser.add_argument('--output', required=True, help='Output CSV file')
    reach_parser.add_argument('--time-periods', type=int, default=39, help='Number of time periods (default: 39)')
    
    # Route command
    route_parser = subparsers.add_parser('route', help='Create route map from CSV')
    route_parser.add_argument('--csv', required=True, help='Input CSV file with routes')
    route_parser.add_argument('--output', default='route_map.html', help='Output HTML file')
    route_parser.add_argument('--blue', action='store_true', help='Use blue color scheme')
    
    # Intel command
    intel_parser = subparsers.add_parser('intel', help='Add intel nodes to existing map')
    intel_parser.add_argument('--html', required=True, help='Input HTML map file')
    intel_parser.add_argument('--csv', required=True, help='Intel CSV file')
    intel_parser.add_argument('--output', help='Output HTML file (default: overwrites input)')
    
    # Viewer command
    viewer_parser = subparsers.add_parser('viewer', help='Run interactive Streamlit viewer')
    
    args = parser.parse_args()
    
    if args.command == 'reach':
        generate_reach_data(args.json, args.output, args.time_periods)
    elif args.command == 'route':
        create_route_map(args.csv, args.output, use_blue_color=args.blue)
    elif args.command == 'intel':
        add_intel_nodes_to_html(args.html, args.csv, args.output)
    elif args.command == 'viewer':
        run_streamlit_viewer()
    else:
        parser.print_help()


if __name__ == "__main__":
    # If run with streamlit, it will call this directly
    # Check if we have command line arguments (CLI mode) or not (Streamlit mode)
    if len(sys.argv) > 1 and sys.argv[1] != 'viewer':
        # CLI mode with specific command
        main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'viewer':
        # Explicit viewer command
        run_streamlit_viewer()
    else:
        # No args - could be Streamlit or just help
        # Try to detect if we're in Streamlit by checking if streamlit module exists
        try:
            import streamlit
            # If we can import streamlit, we're likely being run by it
            run_streamlit_viewer()
        except ImportError:
            # Not streamlit, show help
            main()

