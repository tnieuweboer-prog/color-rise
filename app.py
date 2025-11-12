# app.py — Color Rise (stabiele weergave via PIL)
# Zwart-wit afbeeldingen automatisch inkleuren met OpenCV DNN + veilige beeldafhandeling

import os
import numpy as np
import streamlit as st
from urllib.request import Request, urlopen
from PIL import Image  # <-- voor robuuste weergave

# ========== MODEL PADEN ==========
MODEL_DIR = "models"
PROTO = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
PTS   = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# Mirrors (OpenVINO mirror – stabiel)
URLS = {
    "prototxt": [
        "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_deploy_v2.prototxt"
    ],
    "caffemodel": [
        "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel"
    ],
    "pts_in_hull": [
        "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy"
    ],
}

# ========== DOWNLOAD HELPERS ==========
def _try_download(url: str, dst_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=60) as resp, open(dst_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception:
        return False

def _ensure_files():
    mapping = {"prototxt": PROTO, "caffemodel": MODEL, "pts_in_hull": PTS}
    for key, mirrors in URLS.items():
        dst = mapping[key]
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        for link in mirrors:
            if _try_download(link, dst):
                break

def _models_present():
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in (PROTO, MODEL, PTS))

# ========== MODEL LADEN ==========
@st.cache_resource
def load_net():
    import cv2
    _ensure_files()
    if not _models_present():
        raise FileNotFoundError("Modelbestanden ontbreken. Plaats ze in ./models of upload ze.")
    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
    pts = np.load(PTS).transpose().reshape(2, 313, 1, 1)
    class8 = net.getLayer(net.getLayerId("class8_ab"))
    conv8  = net.getLayer(net.getLayerId("conv8_313_rh"))
    class8.blobs = [pts.astype(np.float32)]
    conv8.blobs  = [np.full((1, 313), 2.606, dtype=np.float32)]
    return net

# ========== KLEURFUNCTIE ==========
def colorize(img_bgr: np.ndarray, net, boost: float = 1.15) -> np.ndarray:
    import cv2
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
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost, 0, 255)
        out_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out_bgr

# ========== VEILIG TONEN: via PIL ==========
def show_bgr(img_bgr: np.ndarray, caption: str):
    """
    Streamlit verwacht RGB; we converteren BGR->RGB met NumPy slicing en geven via PIL weer.
    Dit vermijdt TypeErrors door cv2.cvtColor of channels-kwarg verschillen.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        st.error(f"Onverwacht kanaalformaat bij tonen: shape={img_bgr.shape}")
        return
    img_rgb = img_bgr[:, :, ::-1]  # BGR -> RGB
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_rgb, mode="RGB")
    st.image(pil_img, caption=caption, use_container_width=True)

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Color Rise", page_icon="🎨", layout="centered")
st.title("🎨 Zwart-wit → kleur (OpenCV DNN)")

boost = st.sidebar.slider("Saturatie-boost", 1.0, 1.4, 1.15, 0.01)

# Laad model
try:
    net = load_net()
    st.sidebar.success("Model geladen ✅")
except FileNotFoundError as e:
    st.sidebar.error(str(e))
    st.stop()
except Exception as e:
    st.sidebar.error(f"Kon model niet laden: {e}")
    st.stop()

# ========== UPLOAD + VEILIGE AFBEELDINGSHANDLING ==========
file = st.file_uploader("Upload een **zwart-wit** afbeelding", type=["jpg","jpeg","png","bmp","tiff","webp"])

if file is not None:
    import cv2

    # Lees bytes één keer en decodeer (kan 1, 3 of 4 kanalen zijn)
    data = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if img is None:
        st.error("Kon de afbeelding niet decoderen. Probeer een andere JPG/PNG.")
        st.stop()

    # Normaliseer kanaalformaat naar 3-kanaals BGR
    if img.ndim == 2:  # grijs
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:  # BGRA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        st.error(f"Onverwacht kanaalformaat: {img.shape}")
        st.stop()

    # Beperk extreem grote beelden
    max_side = 2048
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Inkleuren
    with st.spinner("Inkleuren..."):
        colored = colorize(img, net, boost)

    col1, col2 = st.columns(2)
    with col1:
        show_bgr(img, "Origineel")
    with col2:
        show_bgr(colored, "Ingekleurd")

    # Download-knop
    ok, buf = cv2.imencode(".jpg", colored, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if ok:
        st.download_button(
            "Download gekleurde afbeelding",
            data=bytes(buf),
            file_name="colorized.jpg",
            mime="image/jpeg"
        )

st.caption("💡 Tip: modelbestanden ontbreken? Plaats ze in ./models/ of upload ze via de UI.")




