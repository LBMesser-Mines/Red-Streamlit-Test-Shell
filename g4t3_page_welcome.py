# g4t3_page_welcome.py
# Renders the 'Welcome' and 'How G4T3 Works' pages.

import streamlit as st

def render_welcome_page():
    """Displays the main welcome page with project title and team members."""
    st.markdown("")
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        # Assuming g4t3.png is in the root directory of the app
        st.image("g4t3.png", use_container_width=True)
    st.markdown("---")

    st.title("Meet the Team")
    cols = st.columns(4)
    team_info = [
        ("Professors", ["🗽Daniel Bienstock", "⚒️Alexandra Newman"]),
        ("PhD Students", ["🗽Blake Sisson", "⚒️Justin Kilb", "⚒️Luke Messer"]),
        ("Master's Students", ["⚒️Bobby Provine", "⚒️Brandon Werling", "⚒️Caleb Fluker", "⚒️Gabe Hake"]),
        ("Undergraduates", ["⚒️Kevin Bamwisho", "⚒️Steph Shiferaw"]),
    ]
    for col, (title, names) in zip(cols, team_info):
        col.subheader(title)
        for name in names:
            col.write(f"- {name}")

def render_how_it_works_page():
    """Displays the explanation page for the G4T3 model."""
    st.header("How G4T3 Works")
    # Assuming g4t3_loop.png is in the root directory of the app
    st.image("g4t3_loop.png", use_container_width=True)
    st.markdown("**View the math here**")
