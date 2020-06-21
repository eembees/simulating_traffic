import streamlit as st

import src.pages.about
import src.pages.home
import src.pages.tasep
# import src.pages.nasch
import src.utils


PAGES = {
    "Home": src.pages.home,
    "About": src.pages.about,
    "TASEP": src.pages.tasep,
    # "NaSch": src.pages.nasch,
}


def main() -> None:
    """
    Main Function of the streamlit app
    """

    st.sidebar.title("Navigate")
    selection = st.sidebar.radio("Select Page", list(PAGES.keys()))

    curr_page = PAGES[selection]
    with st.spinner(f"Loading {selection} ..."):
        src.utils.write_page(page=curr_page)



if __name__ == "__main__":
    main()
