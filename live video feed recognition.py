# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:00:46 2024

@author: DELL
"""

import torch
import cv2
import numpy as np

# Specify the path to your downloaded yolov5s.pt model
model_path = r'Z:\Download\yolov5s.pt'  # Raw string path to avoid escape character issues

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Start video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is open
if not cap.isOpened():
    print("Error: Could not open video feed.")
else:
    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform inference on the current frame
        results = model(frame)

        # Render the bounding boxes and labels on the frame
        results.render()  # This draws the bounding boxes on the frame in-place

        # Get the processed frame with bounding boxes
        detected_frame = results.ims[0]

        # Convert BGR (OpenCV format) to RGB (for correct display in OpenCV)
        detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)

        # Display the frame with bounding boxes
        cv2.imshow('Live Video Feed - YOLOv5 Detection', detected_frame_rgb)

        # Wait for the user to press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
