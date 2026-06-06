"""
Sterling Vision Package
=======================
Supports two backends — pick one in config.yaml:

  backend: "huskylens"   → core/vision.py   (USB serial, HuskyLens2 hardware)
  backend: "webcam"      → vision/webcam.py  (USB cam, YOLO + face_recognition)

Both expose the same Block interface so main.py works with either.
"""
