#!/bin/bash
set -e

yolo export \
  model=yolov8n-cls.pt \
  format=tflite \
  imgsz=320