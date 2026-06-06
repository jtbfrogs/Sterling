"""
Sterling Vision Package
========================
USB webcam + YOLOv8 object detection + face_recognition for face ID.

Face enrollment:
    Drop a photo into vision/faces/ — filename = person's name.
    vision/faces/jtb.jpg  →  recognised as "jtb"

Config (config.yaml):
    vision:
      enabled: true
      device_index: 0
      yolo_model: "yolov8n.pt"
      face_recognition: true
      known_faces_dir: "vision/faces"
"""
