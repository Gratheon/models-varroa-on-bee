import os
import time
import uuid
import datetime
from typing import Any, Dict

from flask import Flask, jsonify, request

app = Flask(__name__)


def _ts() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def _log(level: str, message: str, **kwargs) -> None:
    details = " ".join([f"{k}={v}" for k, v in kwargs.items()])
    suffix = f" {details}" if details else ""
    print(f"[{_ts()}] [{level}] {message}{suffix}", flush=True)


def _run_detection(image_bytes: bytes, request_id: str) -> Any:
    from detect import run

    weights = os.getenv("MODEL_WEIGHTS", "/app/yolo11n.pt")
    conf_thres = float(os.getenv("CONF_THRES", "0.25"))
    iou_thres = float(os.getenv("IOU_THRES", "0.45"))
    imgsz = int(os.getenv("IMG_SIZE", "640"))
    max_det = int(os.getenv("MAX_DET", "20"))

    _log(
        "INFO",
        "running varroa-on-bee inference",
        request_id=request_id,
        weights=weights,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        imgsz=imgsz,
        max_det=max_det,
        image_bytes=len(image_bytes),
    )

    return run(
        image_buffer=image_bytes,
        weights=weights,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        imgsz=imgsz,
        max_det=max_det,
    )


@app.get("/")
def index():
    return """
    <html>
      <body>
        <h1>Varroa On Bee Detector API</h1>
        <form method="POST" enctype="multipart/form-data">
          <input type="file" name="file" accept="image/*" />
          <input type="submit" value="Upload and Detect" />
        </form>
      </body>
    </html>
    """


@app.get("/health")
def health() -> Dict[str, str]:
    return {"message": "varroa-on-bee detector is running"}


@app.post("/")
def detect_endpoint():
    request_id = str(uuid.uuid4())[:8]
    started_at = time.perf_counter()
    _log(
        "INFO",
        "incoming detect request",
        request_id=request_id,
        path=request.path,
        method=request.method,
        remote_addr=request.remote_addr,
        content_length=request.content_length,
        content_type=request.content_type,
    )

    if "file" not in request.files:
        _log("WARN", "rejecting request, missing file field", request_id=request_id)
        return jsonify({"message": "Missing 'file' in multipart form data", "result": [], "count": 0}), 400

    upload = request.files["file"]
    image_bytes = upload.read()
    _log(
        "INFO",
        "received upload",
        request_id=request_id,
        filename=upload.filename,
        mimetype=upload.mimetype,
        image_bytes=len(image_bytes),
    )

    if not image_bytes:
        _log("WARN", "rejecting request, empty uploaded file", request_id=request_id, filename=upload.filename)
        return jsonify({"message": "Empty file uploaded", "result": [], "count": 0}), 400

    detections = _run_detection(image_bytes, request_id=request_id)
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    _log(
        "INFO",
        "request processed",
        request_id=request_id,
        detections=len(detections),
        duration_ms=duration_ms,
    )

    return jsonify(
        {
            "message": "File processed successfully",
            "result": detections,
            "count": len(detections),
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8752"))
    _log(
        "INFO",
        "starting varroa-on-bee server",
        port=port,
        model_weights=os.getenv("MODEL_WEIGHTS", "/app/yolo11n.pt"),
        conf_thres=os.getenv("CONF_THRES", "0.25"),
        iou_thres=os.getenv("IOU_THRES", "0.45"),
        img_size=os.getenv("IMG_SIZE", "640"),
        max_det=os.getenv("MAX_DET", "20"),
    )
    app.run(host="0.0.0.0", port=port)
