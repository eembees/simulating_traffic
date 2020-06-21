import streamlit as st


about_str = """Written by Magnus Berg Sletfjerding, May and June 2020.\n
Inspired by Namiko Mitarai's course in Diffusive and Stochastic Processes in 2020.\n
Thanks to Namiko for helping with explaining throughout the course as well as afterwards."""

def write():
    st.title("About Page")
    st.write(about_str)
