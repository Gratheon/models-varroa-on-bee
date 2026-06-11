import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from gratheon_log_lib import error, info, warn
from ultralytics import YOLO


# Production CPU does not support NNPACK; leaving it enabled prints scary warnings
# for every inference even though detection succeeds. Disable it before model use.
torch.backends.nnpack.set_flags(False)

_model: Optional[YOLO] = None


def load_model(weights_path: str) -> YOLO:
    global _model
    if _model is None:
        info("loading varroa-on-bee model", {"weights_path": weights_path})
        _model = YOLO(weights_path, verbose=False)
        info("model loaded")
    return _model


def run(
    image_buffer: bytes,
    weights: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    imgsz: int = 640,
    max_det: int = 20,
) -> List[Dict[str, Any]]:
    if not image_buffer:
        warn("empty image buffer passed to detector")
        return []

    started_at = time.perf_counter()
    info(
        "starting detection",
        {
            "image_bytes": len(image_buffer),
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "imgsz": imgsz,
            "max_det": max_det,
        },
    )

    model = load_model(weights)

    nparr = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        error("failed to decode image buffer")
        return []

    info("decoded image", {"shape": image.shape})

    inference_started_at = time.perf_counter()
    results = model(
        image,
        conf=conf_thres,
        iou=iou_thres,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
    )
    inference_ms = round((time.perf_counter() - inference_started_at) * 1000, 2)

    detections: List[Dict[str, Any]] = []
    per_result_counts: List[int] = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            per_result_counts.append(0)
            continue

        per_result_counts.append(len(boxes))

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0]),
                    "class_name": "varroa_on_bee",
                }
            )

    total_ms = round((time.perf_counter() - started_at) * 1000, 2)
    max_conf = round(max((d["confidence"] for d in detections), default=0.0), 4)
    min_conf = round(min((d["confidence"] for d in detections), default=0.0), 4)

    info(
        "detection complete",
        {
            "detections": len(detections),
            "per_result_counts": per_result_counts,
            "min_conf": min_conf,
            "max_conf": max_conf,
            "inference_ms": inference_ms,
            "total_ms": total_ms,
        },
    )

    return detections
