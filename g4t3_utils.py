# g4t3_utils.py
# This module contains shared utility functions used across the application.

import streamlit as st
import pandas as pd

def stage_label(key: str) -> str:
    """
    Converts an internal scenario key (e.g., "M0", "M1") into a
    user-friendly label for display in the UI.
    """
    if not isinstance(key, str) or not key.startswith('M'):
        return str(key)
    
    if key == "M0":
        return "Cornell DC_LOG"
    else:
        try:
            # Safely convert the numeric part of the key
            stage_num = int(key.lstrip('M'))
            return f"Risk Aware Stage {stage_num}"
        except (ValueError, TypeError):
            return str(key) # Fallback for unexpected format

def get_cornell_max_link_weight(all_data):
    """
    Calculates the maximum weight on any single link in the baseline M0 scenario.
    This value is cached to avoid re-computation.
    """
    if "cornell_max_link_weight" not in st.session_state:
        df_corn = pd.DataFrame(all_data["M0"])
        df_corn["weight"] = df_corn["Mvalue"] * df_corn["per-item weight"]
        df_corn["route"]  = df_corn["l"] + " → " + df_corn["lprime"]
        st.session_state["cornell_max_link_weight"] = (
            df_corn.groupby(["t", "route"])["weight"].sum().max()
        )
    return st.session_state["cornell_max_link_weight"]