import streamlit as st
import numpy as np
import cv2
import os
from urllib.request import urlretrieve

MODEL_DIR = "models"
PROTO = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
PTS   = os.path.join(MODEL_DIR, "pts_in_hull.npy")

URLS = {
    "prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/colorization_deploy_v2.prototxt",
    "caffemodel": "https://github.com/richzhang/colorization/blob/caffe_release/models/colorization_release_v2.caffemodel?raw=1",
    "pts_in_hull": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/pts_in_hull.npy",
}

@st.cache_resource
def load_net():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(PROTO): urlretrieve(URLS["prototxt"], PROTO)
    if not os.path.exists(MODEL): urlretrieve(URLS["caffemodel"], MODEL)
    if not os.path.exists(PTS):   urlretrieve(URLS["pts_in_hull"], PTS)

    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
    pts = np.load(PTS).transpose().reshape(2,313,1,1)
    class8 = net.getLayer(net.getLayerId("class8_ab"))
    conv8  = net.getLayer(net.getLayerId("conv8_313_rh"))
    class8.blobs = [pts.astype(np.float32)]
    conv8.blobs  = [np.full((1,313), 2.606, dtype=np.float32)]
    return net

def colorize(img_bgr, net, boost=1.15):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    L = lab[:,:,0]
    L_rs = cv2.resize(L, (224,224)) - 50
    net.setInput(cv2.dnn.blobFromImage(L_rs))
    ab = net.forward()[0].transpose((1,2,0))
    ab_up = cv2.resize(ab, (img_rgb.shape[1], img_rgb.shape[0]))
    out_lab = np.concatenate((L[:,:,None], ab_up), axis=2).astype(np.float32)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
    out_bgr = np.clip(out_bgr, 0, 255).astype(np.uint8)
    if boost and boost != 1.0:
        hsv = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * float(boost), 0, 255)
        out_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out_bgr

st.set_page_config(page_title="Zwart-wit naar kleur", page_icon="🎨")
st.title("🎨 Zwart-wit inkleuren (OpenCV DNN)")
boost = st.slider("Saturatie-boost", 1.0, 1.4, 1.15, 0.01)

net = load_net()

file = st.file_uploader("Upload een zwart-wit afbeelding", type=["jpg","jpeg","png","bmp","tiff","webp"])
if file:
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    colored = colorize(img, net, boost)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Origineel")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    with col2:
        st.subheader("Ingekleurd")
        st.image(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    # Download-knop
    retval, buf = cv2.imencode(".jpg", colored, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    st.download_button("Download gekleurde afbeelding", data=bytes(buf), file_name="colorized.jpg", mime="image/jpeg")
