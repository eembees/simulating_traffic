import streamlit as st


def write():
    st.title("Home page")
    st.write(
        "Here I present a visualizer for different ways to visualize traffic flows using Streamlit. "
        "Currently the only implemented models are the Totally Asymmetrical Exclusion Process (TASEP), "
        "and the TASEP implemented with parallel updates, also known as the NaSch model."
        ""
    )
