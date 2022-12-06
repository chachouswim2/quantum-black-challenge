import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from owslib.wms import WebMapService

import numpy as np
from tensorflow import keras
import keras.utils as image
from PIL import Image

st.set_page_config(layout="wide", page_title="Browse silos in Iowa", page_icon=":pushpin:")
st.title('Browse silos in Iowa')

@st.cache(allow_output_mutation=True)
def load_model():
    model = keras.models.load_model('model3.h5')
    return model
    
model = load_model()
model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

img_width, img_height = 256, 256

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
        coords = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])
    except: 
        coords = (41.579830, -93.791328)

    st.write(coords)

    wms = WebMapService("https://ortho.gis.iastate.edu/arcgis/services/ortho/naip_2021_nc/ImageServer/WMSServer?")
    name = "naip_2021_nc"

    wms.getOperationByName('GetMap').methods[0]["url"] = "https://ortho.gis.iastate.edu/arcgis/services/ortho/naip_2021_nc/ImageServer/WMSServer"
    response = wms.getmap(
        layers=[
            name,
        ],
        # Left, bottom, right, top
        #bbox=(-93.746323, 41.326376, -93.735312, 41.334719),
        bbox=(coords[1]-0.001, coords[0]-0.001, coords[1]+0.001, coords[0]+0.001),
        format="image/png",
        size=(256, 256),
        srs="EPSG:4326",
        transparent=False,
    )

    #L'image à lire est contenue dans le image.read()

    st.image(response.read())

    img = Image.open(response).convert('RGB')

    img = image.img_to_array(img)
    img= img.astype('float32') / 255.
    img = np.expand_dims(img, axis = 0)
    prob = model.predict(img)

    st.write(f"Forecasted probability that the picture contains silo(s): {prob[0][0]*100:.0f}%")
    if prob[0][0]>=0.5:
        st.success(f"Silos have been identified in this picture.", icon="✅")
    else:
        st.error(f"No silos have been identified in this picture.", icon="❌")

    #out = open('avhrr.png', 'wb')
    #out.write(response.read())
    #out.close()