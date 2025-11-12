import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from urllib.request import urlretrieve

MODEL_DIR = "models"
PROTO = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
PTS   = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# Publieke, gangbare locaties van de OpenCV modelbestanden
URLS = {
    "prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/colorization_deploy_v2.prototxt",
    "caffemodel": "https://github.com/richzhang/colorization/blob/caffe_release/models/colorization_release_v2.caffemodel?raw=1",
    "pts_in_hull": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/pts_in_hull.npy",
}

def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(PROTO):
        print("• Downloading prototxt...")
        urlretrieve(URLS["prototxt"], PROTO)
    if not os.path.exists(MODEL):
        print("• Downloading caffemodel (ca. 130 MB)...")
        urlretrieve(URLS["caffemodel"], MODEL)
    if not os.path.exists(PTS):
        print("• Downloading pts_in_hull.npy...")
        urlretrieve(URLS["pts_in_hull"], PTS)

def load_net():
    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

    pts = np.load(PTS)  # (313, 2)
    pts = pts.transpose().reshape(2, 313, 1, 1)

    # Laad cluster-centers in de juiste lagen
    class8 = net.getLayer(net.getLayerId("class8_ab"))
    conv8  = net.getLayer(net.getLayerId("conv8_313_rh"))
    class8.blobs = [pts.astype(np.float32)]
    conv8.blobs  = [np.full((1, 313), 2.606, dtype=np.float32)]

    return net

def colorize_image(img_bgr, net, saturation_boost=1.0):
    if img_bgr is None:
        raise ValueError("Lege of ongeldige afbeelding.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    L = img_lab[:, :, 0]

    L_rs = cv2.resize(L, (224, 224))  # model input
    L_rs = L_rs - 50  # mean-centering

    net.setInput(cv2.dnn.blobFromImage(L_rs))
    ab_dec = net.forward()[0].transpose((1, 2, 0))  # (56, 56, 2)
    ab_up = cv2.resize(ab_dec, (img_rgb.shape[1], img_rgb.shape[0]))

    out_lab = np.concatenate((L[:, :, np.newaxis], ab_up), axis=2).astype(np.float32)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)

    out_bgr = np.clip(out_bgr, 0, 255).astype(np.uint8)

    if saturation_boost and saturation_boost != 1.0:
        hsv = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= float(saturation_boost)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        out_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out_bgr

def process_file(in_path, out_path, net, saturation_boost):
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Kan afbeelding niet lezen: {in_path}")
    out = colorize_image(img, net, saturation_boost=saturation_boost)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)

def is_image(fname):
    ext = os.path.splitext(fname)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]

def main():
    parser = argparse.ArgumentParser(
        description="Zwart-wit automatisch inkleuren met OpenCV DNN."
    )
    parser.add_argument("input", help="Pad naar bestand of map met afbeeldingen")
    parser.add_argument("-o", "--output", default="output",
                        help="Uitvoerbestand of map (default: ./output)")
    parser.add_argument("--boost", type=float, default=1.15,
                        help="Saturatie-boost (1.0 = uit, bv. 1.15 is subtiel)")
    args = parser.parse_args()

    # 1) modellen binnenhalen indien nodig
    try:
        ensure_models()
    except Exception as e:
        print("Kon modellen niet automatisch downloaden.")
        print("Zet ze handmatig in ./models en probeer opnieuw.")
        print(f"Technische melding: {e}")
        sys.exit(1)

    # 2) netwerk laden
    net = load_net()

    # 3) single file of batch
    if os.path.isfile(args.input):
        in_file = args.input
        out_file = args.output
        if os.path.isdir(args.output):
            base = os.path.basename(in_file)
            name, ext = os.path.splitext(base)
            out_file = os.path.join(args.output, f"{name}_color{ext}")
        process_file(in_file, out_file, net, args.boost)
        print(f"Gereed: {out_file}")
    elif os.path.isdir(args.input):
        in_dir = args.input
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
        images = [f for f in os.listdir(in_dir) if is_image(f)]
        if not images:
            print("Geen afbeeldingen gevonden in de map.")
            sys.exit(1)
        for f in tqdm(images, desc="Inkleur-progress"):
            in_file = os.path.join(in_dir, f)
            name, ext = os.path.splitext(f)
            out_file = os.path.join(out_dir, f"{name}_color{ext}")
            try:
                process_file(in_file, out_file, net, args.boost)
            except Exception as e:
                print(f"Overgeslagen {f}: {e}")
        print(f"Klaar. Resultaat in map: {out_dir}")
    else:
        print("Input is geen bestand of map.")
        sys.exit(1)

if __name__ == "__main__":
    main()
