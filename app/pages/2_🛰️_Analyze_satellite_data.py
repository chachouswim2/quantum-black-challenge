import streamlit as st

st.set_page_config(layout="wide", page_title="Upload a satellite image", page_icon=":satellite:")
st.title('Upload a satellite image')


upload = st.file_uploader("Upload your file here:")