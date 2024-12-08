import streamlit as st

import constants
import streamlit_pages.home_page
import streamlit_pages.eda_page
import streamlit_pages.models_page
from streamlit_option_menu import option_menu

# Streamlit app layout
def main():

    with st.sidebar:
        selected = option_menu(
            menu_title='Navigation',
            options=[constants.HOME_PAGE_NAME, constants.EDA_PAGE_NAME, constants.MODELS_PAGE_NAME],
            icons=['house', 'graph-up-arrow', 'diagram-3'],
            menu_icon='map',
            default_index=0
        )

    if selected == constants.HOME_PAGE_NAME:
        streamlit_pages.home_page.display_home_page()

    if selected == constants.EDA_PAGE_NAME:
        streamlit_pages.eda_page.display_eda_page()

    if selected == constants.MODELS_PAGE_NAME:
        streamlit_pages.models_page.display_models_page()

if __name__ == "__main__":
    main()
