import streamlit as st


about_str = """Written by Magnus Berg Sletfjerding, May and June 2020.\n
Inspired by Namiko Mitarai's course in Diffusive and Stochastic Processes in 2020.\n
Thanks to Namiko for helping with explaining throughout the course as well as afterwards.\n
All the equations and calculations are based on her Lecture Notes.
"""
link_md = """The NaSch paper can be found [here](https://doi.org/10.1051/jp1:1992277).  
Source code at [**GitHub**](https://github.com/eembees/simulating_traffic)"""


def write():
    st.title("About Page")
    st.write(about_str)
    st.markdown(link_md)
