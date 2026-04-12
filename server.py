import logging
import os
import time
import uuid
from typing import Any, Dict

from flask import Flask, Response, g, jsonify, request
from gratheon_log_lib import bind_context, clear_context, configure, error_enriched, info, warn
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
configure()


class LogLibHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        meta = {
            "logger_name": record.name,
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }
        if record.levelno >= logging.ERROR:
            if record.exc_info:
                error_enriched(message, record.exc_info[1], meta)
            else:
                warn(message, meta)
            return
        if record.levelno >= logging.WARNING:
            warn(message, meta)
            return
        info(message, meta)


def _configure_framework_logging() -> None:
    handler = LogLibHandler()
    handler.setLevel(logging.INFO)

    for logger_name in ("werkzeug", "flask.app"):
        framework_logger = logging.getLogger(logger_name)
        framework_logger.handlers.clear()
        framework_logger.propagate = False
        framework_logger.setLevel(logging.INFO)
        framework_logger.addHandler(handler)

    app.logger.handlers.clear()
    app.logger.propagate = False
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(handler)


_configure_framework_logging()


def _run_detection(image_bytes: bytes, request_id: str) -> Any:
    from detect import run

    weights = os.getenv("MODEL_WEIGHTS", "/app/yolo11n.pt")
    conf_thres = float(os.getenv("CONF_THRES", "0.25"))
    iou_thres = float(os.getenv("IOU_THRES", "0.45"))
    imgsz = int(os.getenv("IMG_SIZE", "640"))
    max_det = int(os.getenv("MAX_DET", "20"))

    info(
        "running varroa-on-bee inference",
        {
            "request_id": request_id,
            "weights": weights,
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "imgsz": imgsz,
            "max_det": max_det,
            "image_bytes": len(image_bytes),
        },
    )

    return run(
        image_buffer=image_bytes,
        weights=weights,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        imgsz=imgsz,
        max_det=max_det,
    )


@app.before_request
def before_request() -> None:
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    g.request_started_at = time.perf_counter()
    g.request_id = request_id
    bind_context(request_id=request_id)
    info(
        "request started",
        {
            "path": request.path,
            "method": request.method,
            "remote_addr": request.remote_addr,
            "content_length": request.content_length,
            "content_type": request.content_type,
            "user_agent": request.user_agent.string,
        },
    )


@app.after_request
def after_request(response: Response) -> Response:
    started_at = getattr(g, "request_started_at", None)
    duration_ms = None
    if started_at is not None:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

    info(
        "request finished",
        {
            "path": request.path,
            "method": request.method,
            "status_code": response.status_code,
            "content_length": response.calculate_content_length(),
            "duration_ms": duration_ms,
        },
    )
    clear_context()
    return response


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    if isinstance(exc, HTTPException):
        return exc
    error_enriched(
        "unhandled request error",
        exc,
        {
            "path": request.path,
            "method": request.method,
        },
    )
    return jsonify({"message": "Internal server error", "result": [], "count": 0}), 500


@app.teardown_request
def teardown_request(_exc: Exception | None) -> None:
    clear_context()


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
    started_at = time.perf_counter()
    info(
        "incoming detect request",
        {
            "path": request.path,
            "method": request.method,
            "remote_addr": request.remote_addr,
            "content_length": request.content_length,
            "content_type": request.content_type,
        },
    )

    if "file" not in request.files:
        warn("rejecting request, missing file field")
        return jsonify({"message": "Missing 'file' in multipart form data", "result": [], "count": 0}), 400

    upload = request.files["file"]
    image_bytes = upload.read()
    info(
        "received upload",
        {
            "filename": upload.filename,
            "mimetype": upload.mimetype,
            "image_bytes": len(image_bytes),
        },
    )

    if not image_bytes:
        warn("rejecting request, empty uploaded file", {"filename": upload.filename})
        return jsonify({"message": "Empty file uploaded", "result": [], "count": 0}), 400

    detections = _run_detection(image_bytes, request_id=g.request_id)
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    info(
        "request processed",
        {
            "detections": len(detections),
            "duration_ms": duration_ms,
        },
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
    info(
        "starting varroa-on-bee server",
        {
            "port": port,
            "model_weights": os.getenv("MODEL_WEIGHTS", "/app/yolo11n.pt"),
            "conf_thres": os.getenv("CONF_THRES", "0.25"),
            "iou_thres": os.getenv("IOU_THRES", "0.45"),
            "img_size": os.getenv("IMG_SIZE", "640"),
            "max_det": os.getenv("MAX_DET", "20"),
        },
    )
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
