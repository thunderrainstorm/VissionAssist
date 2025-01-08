from gpiozero import DistanceSensor, AngularServo
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Initialize distance sensor and servo motor with max_distance set to 4 meters
ultrasonic_sensor = DistanceSensor(echo=17, trigger=4, max_distance=4)  # Adjust GPIO pins as necessary
servo = AngularServo(16, min_angle=-90, max_angle=90)  # Adjust GPIO pin and angles as necessary

# Set up object detection tracking
detected_objects = set()  
previously_detected = set()  
announcement_delay = 5  
last_announcement_time = time.time()  

url = "http://192.168.22.238:8080/video"  # mobile cam URL
cap = cv2.VideoCapture(url)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("/home/Tejas/exdark/yolov11m.pt")  # Load YOLO model

classNames = [
    "people", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "table", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

prev_frame_time = 0
new_frame_time = 0

# Screen width for determining object position
screen_width = 1280

def measure_distance(direction):
    """Rotate servo and measure distance based on the object's direction."""
    if direction == "left":
        servo.angle = -60
    elif direction == "right":
        servo.angle = 60
    else:
        servo.angle = 0  # center position

    # Increase delay to allow servo motor to fully rotate to its position
    time.sleep(0.5)  # Adjust this delay as needed based on the servo speed

    # Measure the distance after the servo has reached its position
    distance = ultrasonic_sensor.distance * 100  # Convert to centimeters
    return round(distance, 1)

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera.")
        break
    
    results = model(img, stream=True)
    current_detected_objects = set()  # Reset current detected objects for this frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Only process if confidence is greater than 0.3
            if conf > 0.3:
                # Determine the direction of the detected object
                object_center_x = (x1 + x2) / 2
                if object_center_x < screen_width / 3:
                    direction = "left"
                elif object_center_x > 2 * screen_width / 3:
                    direction = "right"
                else:
                    direction = "center"

                # Add to the current detected objects
                current_detected_objects.add((class_name, direction))

                display_text = f'{class_name} {conf} {direction}'
                cvzone.putTextRect(img, display_text, (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Check for new objects that have entered the frame
    for obj, direction in current_detected_objects:
        if obj not in detected_objects and time.time() - last_announcement_time >= announcement_delay:
            # Measure the distance in the specified direction
            distance = measure_distance(direction)
            
            # Only announce if the object is less than 3.99 meters away
            if distance < 399:  # 3.99 meters in centimeters
                announcement = f"{obj} detected on the {direction}, distance is {distance} centimeters"
                engine.say(announcement)
                engine.runAndWait()
                detected_objects.add(obj)
                last_announcement_time = time.time()

    # Check for objects that have exited the frame
    for obj in list(detected_objects):
        if obj not in [o[0] for o in current_detected_objects]:  # Only the class name
            previously_detected.add(obj)
            detected_objects.remove(obj)

    # Update FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()
