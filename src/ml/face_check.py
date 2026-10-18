import io

import cv2
import numpy as np
from PIL import Image, ImageOps

_MAX_SIDE = 1280
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def prepare(data):
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > _MAX_SIDE:
        img.thumbnail((_MAX_SIDE, _MAX_SIDE))
    return np.asarray(img)


def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    found = _CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in found]


def annotate(img, boxes):
    arr = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(arr, (x, y), (x + w, y + h), (255, 76, 47), 2)
    return Image.fromarray(arr)
