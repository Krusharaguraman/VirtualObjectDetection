import cv2
import torch
from yolov5 import YOLOv5

# Load the YOLOv5 model (pre-trained)
model = YOLOv5("yolov5s.pt")  # You can change to yolov5m.pt, yolov5l.pt, yolov5x.pt for larger models

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Perform object detection using the `predict` method
    results = model.predict(frame)

    # Render the results (bounding boxes and labels)
    frame = results.render()[0]

    # Display the resulting frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
