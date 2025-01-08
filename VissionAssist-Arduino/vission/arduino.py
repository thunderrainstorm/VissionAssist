import cv2
import cvzone
import math
import pyttsx3
import time
import serial
from ultralytics import YOLO

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[1].id)
tts_engine.setProperty('rate', 150)

# Initialize serial communication
arduino = serial.Serial('COM3', 9600)  # Adjust COM port as needed

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    results = model(img)
    current_boxes = []

    # Detect objects and calculate centroids
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cls = int(box.cls[0])  # Get the class of the detected object

            current_boxes.append((x1, y1, x2, y2, cls))  # Include class in the box
            conf = math.ceil((box.conf[0] * 100)) / 100

            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    time.sleep(0.5)
    # Read direction and distance from Arduino
    if arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8').strip()
        if line:
            # Check if the line contains the expected number of values
            values = line.split(',')
            if len(values) == 2:  # Check if we have both direction and distance
                direction, distance = values
                print(f"Object detected at {distance} cm to the {direction}")

                # Speech synthesis for each detected object
                for box in current_boxes:
                    detected_object = classNames[box[4]]  # Use the cls from the box
                    announcement = f"{detected_object} detected at {distance} centimeters to the {direction}"
                    tts_engine.say(announcement)
                    tts_engine.runAndWait()
            else:
                print(f"Received unexpected data from Arduino: {line}")

    # Display the video feed
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
