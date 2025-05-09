#!/usr/bin/env python
# coding: utf-8

# In[11]:


import logging
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pyttsx3


# In[12]:


tts_engine = pyttsx3.init()

voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[1].id) 
tts_engine.setProperty('rate', 150)


# In[17]:


#To avoid YOLO from logging unnecessarily. Prevents cluttering
logging.getLogger("ultralytics").setLevel(logging.WARNING)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  #width
cap.set(4, 720)   #height

model = YOLO("../Yolo-Weights/yolov8n.pt")

#coco classes
with open("coco.names", "r") as coco_name_file:
    classNames = [line.strip() for line in coco_name_file.readlines()]
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"]


# In[18]:


# def get_centroid(box):
#     x1, y1, x2, y2 = box
#     cx = (x1+x2)//2
#     cy = (y1+y2)//2
#     return (cx, cy)

# def centroid_distance(box1, box2):
#     cx1, cy1 = get_centroid(box1)
#     cx2, cy2 = get_centroid(box2)
#     return math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)

# def is_new_detection(box, previous_boxes, distance_threshold=100):
#     for prev_box in previous_boxes:
#         if centroid_distance(box, prev_box) < distance_threshold:
#             return False
#     return True

def iou(box1, box2):
    """Function to calculate Intersection over Union (IoU) between two boxes"""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def is_new_detection(box, previous_boxes, threshold=0.3):
    for prev_box in previous_boxes:
        if iou(box, prev_box) > threshold:
            return False
    return True


# In[19]:


previous_boxes = []
limit = 3
while True:
    new_frame_time = time.time()
    success, img = cap.read()

    results = model(img)

    current_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            current_box = (x1, y1, x2, y2)

            if is_new_detection(current_box, previous_boxes):
                current_boxes.append(current_box)
                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                print(f"{classNames[cls]} with confidence {conf}")
                announcement = f"{classNames[cls]} detected"
                tts_engine.say(announcement)
                tts_engine.runAndWait()

    previous_boxes.extend(current_boxes)
    if(len(previous_boxes) > limit):
        previous_boxes = previous_boxes[-limit:-1]
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




