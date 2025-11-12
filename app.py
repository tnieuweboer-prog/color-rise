import os
import io
import numpy as np
import cv2
import streamlit as st
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

MODEL_DIR = "models"
PROTO = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
PTS   = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# Meerdere mirrors voor robuustheid
URLS = {
    "prototxt": [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/colorization_deploy_v2.prototxt",
        "https://github.com/opencv/opencv/raw/master/samples/dnn/colorization_deploy_v2.prototxt",
    ],
    "caffemodel": [
        "https://github.com/richzhang/colorization/raw/caffe_release/models/colorization_release_v2.caffemodel",
        "https://raw.githubusercontent.com/richzhang/colorization/caffe_release/models/colorization_release_v2.caffemodel",
    ],
    "pts_in_hull": [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/pts_in_hull.npy",
        "https://github.com/opencv/opencv/raw/master/samples/dnn/pts_in_hull.npy",
    ],
}

def _try_download(url: str, dst_path: str, timeout: int = 30) -> bool:
    """Download met User-Agent header en time-out. Return True bij succes."""
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as resp, open(dst_path, "wb") as f:
            f.write(resp.read())
        return True
    except (HTTPError, URLError, TimeoutError, ConnectionError) as e:
        return False

def _ensure_file(name: str, paths: list[str], target: str) -> bool:
    """Probeer alle mirrors. Return True als bestand aanwezig of download gelukt."""
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return True
    for u in paths:
        if _try_download(u, target):
            return True
    return False

def _models_present() -> bool:
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in (PROTO, MODEL, PTS))

def _offer_manual_upload():
    st.warning("Kon de modelbestanden niet downloaden. Upload ze hieronder of plaats ze in de map ./models.")
    with st.expander("Modelbestanden uploaden (3 stuks)", expanded=True):
        up_prototxt = st.file_uploader("Upload colorization_deploy_v2.prototxt", type=["prototxt"], key="up_proto")
        up_caffemodel = st.file_uploader("Upload colorization_release_v2.caffemodel", type=["caffemodel"], key="up_model")
        up_pts = st.file_uploader("Upload pts_in_hull.npy", type=["npy"], key="up_pts")

        saved = False
        if st.button("Sla uploads op"):
            os.makedirs(MODEL_DIR, exist_ok=True)
            if up_prototxt:
                open(PROTO, "wb").write(up_prototxt.read())
                saved = True
            if up_caffemodel:
                open(MODEL, "wb").write(up_caffemodel.read())
                saved = True
            if up_pts:
                open(PTS, "wb").write(up_pts.read())
                saved = True
            if saved:
                st.success("Bestanden opgeslagen in ./models. Herlaad de app of ga verder.")
            else:
                st.info("Nog niets geüpload.")

@st.cache_resource
def load_net():
    # OFFLINE modus (zet env var OFFLINE=1 om download te skippen)
    offline = os.environ.get("OFFLINE", "0") == "1"

    if not offline:
        _ensure_file("prototxt", URLS["prototxt"], PROTO)
        _ensure_file("caffemodel", URLS["caffemodel"], MODEL)
        _ensure_file("pts_in_hull", URLS["pts_in_hull"], PTS)

    # Als er na downloaden nog steeds niets is, geef gebruiker de kans om te uploaden
    if not _models_present():
        # In Streamlit context kunnen we niet interactief terugkomen uit cache_resource,
        # dus we gooien een exception die we buiten de call afvangen
        raise FileNotFoundError("Modelbestanden ontbreken. Upload vereist.")

    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
    pts = np.load(PTS)  # (313, 2)
    pts = pts.transpose().reshape(2, 313, 1, 1)

    # Laad cluster centers in de juiste lagen
    class8 = net.getLayer(net.getLayerId("class8_ab"))
    conv8  = net.getLayer(net.getLayerId("conv8_313_rh"))
    class8.blobs = [pts.astype(np.float32)]
    conv8.blobs  = [np.full((1, 313), 2.606, dtype=np.float32)]

    return net

def colorize(img_bgr, net, boost=1.15):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    L = lab[:, :, 0]
    L_rs = cv2.resize(L, (224, 224)) - 50
    net.setInput(cv2.dnn.blobFromImage(L_rs))
    ab = net.forward()[0].transpose((1, 2, 0))
    ab_up = cv2.resize(ab, (img_rgb.shape[1], img_rgb.shape[0]))
    out_lab = np.concatenate((L[:, :, None], ab_up), axis=2).astype(np.float32)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
    out_bgr = np.clip(out_bgr, 0, 255).astype(np.uint8)

    if boost and boost != 1.0:
        hsv = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(boost), 0, 255)
        out_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out_bgr

