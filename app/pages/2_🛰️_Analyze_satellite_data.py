import streamlit as st
import numpy as np
from tensorflow import keras
import keras.utils as image
from PIL import Image

st.set_page_config(layout="wide", page_title="Upload a satellite image", page_icon=":satellite:")
st.title('Upload a satellite image')

@st.cache(allow_output_mutation=True)
def load_model():
    model = keras.models.load_model('model3.h5')
    return model
    
model = load_model()
model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

img_width, img_height = 256, 256

upload = st.file_uploader("Upload your file(s) here:", type=['png','jpg'], accept_multiple_files=True)

if upload is not None:

    for uploaded_file in upload:

        img = Image.open(uploaded_file)
        img = img.resize((img_width, img_height))

        c1, c2 = st.columns((1, 2))

        with c1:
            st.image(img)

        #out = open('temp.png', 'wb')
        #out.write(Image.open(upload))
        #out.close()

        with c2:

            with st.spinner('Wait for it...'):

                #img = image.load_img('temp.png', target_size = (img_width, img_height))
                img = image.img_to_array(img)
                img= img.astype('float32') / 255.
                img = np.expand_dims(img, axis = 0)
                prob = model.predict(img)

                st.write(f"Forecasted probability that the picture contains silo(s): {prob[0][0]*100:.0f}%")
                if prob[0][0]>=0.5:
                    st.success(f"Silos have been identified in this picture.", icon="✅")
                else:
                    st.error(f"No silos have been identified in this picture.", icon="❌")
