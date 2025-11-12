# app.py
# Streamlit-app: zwart-wit afbeeldingen automatisch inkleuren met OpenCV DNN
# - Robuuste downloads (meerdere mirrors + User-Agent)
# - Offline fallback: upload de 3 modelbestanden via de UI
# - Lazy imports: cv2 wordt pas geladen wanneer nodig (voorkomt "leeg scherm" bij import errors)

import os
import io
import numpy as np
import streamlit as st
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# -------------------------------
# Pad-configuratie voor modellen
# -------------------------------
MODEL_DIR = "models"
PROTO = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
PTS   = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# Meerdere mirrors voor robuuste downloads
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

# -------------------------------
# Hulpfuncties: download & checks
# -------------------------------
def _try_download(url: str, dst_path: str, timeout: int = 30) -> bool:
    """Download een bestand met User-Agent header. Return True bij succes."""
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as resp, open(dst_path, "wb") as f:
            f.write(resp.read())
        return True
    except (HTTPError, URLError, TimeoutError, ConnectionError):
        return False

def _ensure_from_mirrors(url_list, target) -> bool:
    """Loop alle mirrors langs totdat download lukt."""
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return True
    for u in url_list:
        if _try_download(u, target):
            return True
    return False

def _models_present() -> bool:
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in (PROTO, MODEL, PTS))

def _offer_manual_upload():
    st.warning("Kon de modelbestanden niet downloaden. Upload ze hieronder of plaats ze in ./models.")
    with st.expander("Modelbestanden uploaden (3 stuks)", expanded=True):
        up_prototxt = st.file_uploader("Upload colorization_deploy_v2.prototxt", type=["prototxt"], key="up_proto")
        up_caffemodel = st.file_uploader("Upload colorization_release_v2.caffemodel", type=["caffemodel"], key="up_model")
        up_pts = st.file_uploader("Upload pts_in_hull.npy", type=["npy"], key="up_pts")

        if st.button("Sla uploads op"):
            os.makedirs(MODEL_DIR, exist_ok=True)
            saved_any = False
            if up_prototxt:
                open(PROTO, "wb").write(up_prototxt.read()); saved_any = True
            if up_caffemodel:
                open(MODEL, "wb").write(up_caffemodel.read()); saved_any = True
            if up_pts:
                open(PTS, "wb").write(up_pts.read()); saved_any = True

            if saved_any:
                st.success("Bestanden opgeslagen in ./models. Klik hieronder op 'Probeer opnieuw laden'.")
            else:
                st.info("Nog geen bestand geüpload.")

# -------------------------------
# Model laden (gecached)
# -------------------------------
@st.cache_resource
def load_net(offline: bool = False):
    """
    Laadt het OpenCV colorization-netwerk.
    - offline=True: sla downloads over en verwacht dat bestanden al lokaal staan of via upload komen.
    - offline=False: probeer eerst te downloaden via mirrors, val anders terug op upload.
    """
    # Lazy import: voorkomt crash vóór UI als cv2 nog niet goed geïnstalleerd is
    import cv2

    if not offline:
        _ensure_from_mirrors(URLS["prototxt"], PROTO)
        _ensure_from_mirrors(URLS["caffemodel"], MODEL)
        _ensure_from_mirrors(URLS["pts_in_hull"], PTS)

    if not _models_present():
        # Signaal aan de UI om upload aan te bieden
        raise FileNotFoundError("Modelbestanden ontbreken. Upload vereist.")

    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

    # Laad cluster-centers (313 ab-kleuren) in de juiste lagen
    pts = np.load(PTS)  # shape (313, 2)
    pts = pts.transpose().reshape(2, 313, 1, 1)

    class8 = net.getLayer(net.getLayerId("class8_ab"))
    conv8  = net.getLayer(net.getLayerId("conv8_313_rh"))
    class8.blobs = [pts.astype(np.float32)]
    conv8.blobs  = [np.full((1, 313), 2.606, dtype=np.float32)]

    return net

# -------------------------------
# Kleurfunctie
# -------------------------------
def colorize(img_bgr: np.ndarray, net, boost: float = 1.15) -> np.ndarray:
    import cv2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    L = lab[:, :, 0]
    L_rs = cv2.resize(L, (224, 224)) - 50
    net.setInput(cv2.dnn.blobFromImage(L_rs))
    ab = net.forward()[0].transpose((1, 2, 0))              # (56, 56, 2)
    ab_up = cv2.resize(ab, (img_rgb.shape[1], img_rgb.shape[0]))
    out_lab = np.concatenate((L[:, :, None], ab_up), axis=2).astype(np.float32)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
    out_bgr = np.clip(out_bgr, 0, 255).astype(np.uint8)

    if boost and boost != 1.0:
        hsv = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(boost), 0, 255)
        out_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out_bgr

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Color Rise", page_icon="🎨", layout="centered")
st.title("🎨 Zwart-wit → kleur (OpenCV DNN)")

with st.sidebar:
    st.header("Instellingen")
    offline_mode = st.toggle("Offline modus (niet downloaden)", value=False, help="Zet aan als je host geen internet heeft of downloads blokkeert.")
    boost = st.slider("Saturatie-boost", 1.0, 1.4, 1.15, 0.01, help="Maak kleuren iets levendiger.")

# Probeer het model te laden
net = None
try:
    net = load_net(offline=offline_mode)
except FileNotFoundError:
    _offer_manual_upload()
    if st.button("Probeer opnieuw laden"):
        try:
            net = load_net(offline=True)  # na upload altijd offline herladen
            st.success("Model geladen.")
        except FileNotFoundError:
            st.error("Modelbestanden nog steeds niet gevonden. Upload alle drie de bestanden.")
except Exception as e:
    st.error(f"Kon het model niet initialiseren: {e}")
    st.stop()

# Bestandsuploader
file = st.file_uploader("Upload een **zwart-wit** afbeelding", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

if net is not None and file:
    import cv2
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Kon de afbeelding niet lezen.")
    else:
        with st.spinner("Inkleuren..."):
            colored = colorize(img, net, boost)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Origineel")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("Ingekleurd")
            st.image(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), use_container_width=True)

        ok, buf = cv2.imencode(".jpg", colored, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            st.download_button(
                "Download gekleurde afbeelding",
                data=bytes(buf),
                file_name="colorized.jpg",
                mime="image/jpeg"
            )

st.caption("Tip: werkt het downloaden niet? Zet offline modus aan en upload de 3 modelbestanden via het paneel.")


