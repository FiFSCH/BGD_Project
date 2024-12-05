import streamlit as st
import constants

def display_home_page():
    st.title("Emotion Analysis with PySpark")
    st.image('./images/hugging_face_logo.png')

    st.header("Idea:")
    st.markdown(constants.PROJECT_GOAL)

    st.header("Authors:")
    st.markdown(constants.AUTHORS_LIST)

    st.header('Libraries and Technologies Used:')
    st.markdown(constants.LIBRARIES_USED)