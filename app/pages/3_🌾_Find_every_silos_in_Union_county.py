import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from owslib.wms import WebMapService

import numpy as np
from tensorflow import keras
import keras.utils as image
from PIL import Image

st.set_page_config(layout="wide", page_title="Find every silos in Union county", page_icon=":ear_of_rice:")
st.title('Find every silos in Union county')

