import io
import unittest
import base64
import os
from pathlib import Path
from unittest.mock import patch

from server import app


class ServerTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _jpeg_bytes(self) -> bytes:
        # 1x1 white JPEG
        return base64.b64decode(
            b"/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEA8QEA8QDw8PDw8PDw8PDw8QFREWFhURExUYHSggGBolGxUVITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OGxAQGi0fHyUtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAAEAAQMBIgACEQEDEQH/xAAWAAEBAQAAAAAAAAAAAAAAAAAAAQL/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAGmAP/EABQQAQAAAAAAAAAAAAAAAAAAACD/2gAIAQEAAQUCl//EABQRAQAAAAAAAAAAAAAAAAAAACD/2gAIAQMBAT8Bp//EABQRAQAAAAAAAAAAAAAAAAAAACD/2gAIAQIBAT8Bp//Z"
        )

    def test_returns_400_when_file_missing(self):
        response = self.client.post("/", data={})
        self.assertEqual(response.status_code, 400)

    def test_get_root_serves_upload_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.data.decode("utf-8")
        self.assertIn("<form", body)
        self.assertIn("name=\"file\"", body)

    @patch("server._run_detection")
    def test_detect_endpoint_returns_detections(self, mock_run_detection):
        mock_run_detection.return_value = [
            {
                "x1": 10,
                "y1": 12,
                "x2": 18,
                "y2": 20,
                "confidence": 0.91,
                "class": 0,
                "class_name": "varroa_on_bee",
            }
        ]

        response = self.client.post(
            "/",
            data={"file": (io.BytesIO(self._jpeg_bytes()), "bee.jpg")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(len(body["result"]), 1)

    def test_detect_endpoint_real_model_detects_varroa_from_fixture(self):
        fixture = Path(__file__).parent / "tests" / "fixtures" / "varroa-positive.jpg"
        weights = Path(__file__).parent / "yolo11n.pt"

        self.assertTrue(fixture.exists(), f"Missing fixture image: {fixture}")
        self.assertTrue(weights.exists(), f"Missing model weights: {weights}")

        prev_model_weights = os.environ.get("MODEL_WEIGHTS")
        prev_conf = os.environ.get("CONF_THRES")

        try:
            os.environ["MODEL_WEIGHTS"] = str(weights.resolve())
            os.environ["CONF_THRES"] = "0.05"

            response = self.client.post(
                "/",
                data={"file": (io.BytesIO(fixture.read_bytes()), "varroa-positive.jpg")},
                content_type="multipart/form-data",
            )
        finally:
            if prev_model_weights is None:
                os.environ.pop("MODEL_WEIGHTS", None)
            else:
                os.environ["MODEL_WEIGHTS"] = prev_model_weights
            if prev_conf is None:
                os.environ.pop("CONF_THRES", None)
            else:
                os.environ["CONF_THRES"] = prev_conf

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIsNotNone(body)
        self.assertGreater(body["count"], 0, "Expected at least one varroa detection from fixture image")


if __name__ == "__main__":
    unittest.main()
