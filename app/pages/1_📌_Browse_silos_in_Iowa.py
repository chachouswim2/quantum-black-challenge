import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from owslib.wms import WebMapService

st.set_page_config(layout="wide", page_title="Browse silos in Iowa", page_icon=":pushpin:")
st.title('Browse silos in Iowa')

def get_pos(lat,lng):
    return lat,lng

c1, c2 = st.columns((2, 1))

with c2:
    lng = st.slider('longitude', -96.746323, -90.035312, -93.791328)
    lat = st.slider('latitude', 40.326376, 43.624719, 41.579830)

with c1:
    m = fl.Map(location=[lat,lng])
    m.add_child(fl.LatLngPopup())
    map = st_folium(m, height=500, width=700)

with c2:
    try:
        data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])
    except: 
        data = (41.579830, -93.791328)

    st.write(data)

    wms = WebMapService("https://ortho.gis.iastate.edu/arcgis/services/ortho/naip_2021_nc/ImageServer/WMSServer?")
    name = "naip_2021_nc"

    wms.getOperationByName('GetMap').methods[0]["url"] = "https://ortho.gis.iastate.edu/arcgis/services/ortho/naip_2021_nc/ImageServer/WMSServer"
    response = wms.getmap(
        layers=[
            name,
        ],
        # Left, bottom, right, top
        #bbox=(-93.746323, 41.326376, -93.735312, 41.334719),
        bbox=(data[1]-0.005, data[0]-0.005, data[1]+0.005, data[0]+0.005),
        format="image/png",
        size=(256, 256),
        srs="EPSG:4326",
        transparent=False,
    )

    st.image(response.read())