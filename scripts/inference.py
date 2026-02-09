from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from scripts.viz import colorize_groups, overlay

from scripts.losses_metrics import MeanIoUArgmax, dice_loss_sparse


DEFAULT_SIZE_HW: Tuple[int, int] = (256, 256)
DEFAULT_MODEL_PATH = Path("models") / "best_model.keras"

_MODEL = None


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    global _MODEL
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                "MeanIoUArgmax": MeanIoUArgmax,
                "dice_loss_sparse": dice_loss_sparse,
            },
            compile=False,
        )
    return _MODEL


def preprocess_pil(
    img: Image.Image, size_hw: Tuple[int, int] = DEFAULT_SIZE_HW
) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((size_hw[1], size_hw[0]), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def predict_from_pil(
    img: Image.Image,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    size_hw: Tuple[int, int] = DEFAULT_SIZE_HW,
    alpha: float = 0.45,
):

    model = load_model(model_path)

    x = preprocess_pil(img, size_hw=size_hw)
    pred = model.predict(x[None, ...], verbose=0)[0]

    mask = np.argmax(pred, axis=-1).astype(np.uint8)
    resized_rgb = Image.fromarray((x * 255).astype(np.uint8), mode="RGB")

    mask_color = colorize_groups(mask)
    overlay_img = overlay(resized_rgb, mask_color, alpha=alpha)

    return mask, mask_color, overlay_img, resized_rgb
