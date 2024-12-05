import streamlit as st
import constants

def display_eda_page():
    st.title('Exploratory data analysis:')
    st.markdown(constants.DATA_SET_DESCTIPTION)

    st.header('Distribution of Emotions: ')
    st.image('./images/label_dist.png')

    st.header('Word Length per Emotion: ')
    st.image('./images/length_per_label.png')

    st.header('Most Commonly Occurring Words per Emotion: ')
    st.image('./images/word_cloud_Sadness.png')
    st.image('./images/word_cloud_Joy.png')
    st.image('./images/word_cloud_Love.png')
    st.image('./images/word_cloud_Anger.png')
    st.image('./images/word_cloud_Fear.png')
    st.image('./images/word_cloud_Surprise.png')


