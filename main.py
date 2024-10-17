from ultralytics import YOLO as yolo
from sort import *
import numpy as np
import cv2 as cv
import cvzone
import math

import os

print(os.path.exists("IMG/1.png"))



 
 
model = yolo("../Yolo-Weights/yolov8n.pt")
 
obj = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


clr = {"red":(0,0,255), "purple":(255,0,255),"blue":(255,0,0),"green":(0,255,0)}






def detect(source):
    cap = cv.VideoCapture(source)
    if source == 0:
        cap.set(3, 1280)
        cap.set(4, 720)
    while True:
        _, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(img, (x1, y1, w, h),l=15,rt=1)
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
    
                cvzone.putTextRect(img, f'{obj[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    
        cv.imshow("Image", img)
        k = cv.waitKey(1)
        if k == ord('q'):
            cap.release()
            cv.destroyAllWindows()



def tracker():
    # tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    cap = cv.VideoCapture("VIDEOS/cars.mp4")
    mask = cv.imread("IMG/carmask.png",)
    limits = [50, 450, 673, 450]
    Count = []

    while True:
        _, img = cap.read()

        frame = cv.bitwise_and(img, mask)

        results = model(frame, stream=True)
        detect = np.empty((0,5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                current = obj[cls]
                if current == "car" or current == "truck" or current == "bus" or current == "motorbike" and conf > 0.3:

                    # cvzone.cornerRect(img, (x1, y1, w, h),l=9,rt=5)
                    # cvzone.putTextRect(img, f'{current} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,offset=3)
                    currArr = np.array([x1, y1, x2, y2, conf])
                    detect = np.vstack((detect, currArr))


        resTrack = tracker.update(detect)
        cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), clr["red"], 5)

        for res in resTrack:
            x1, y1, x2, y2, track_id = map(int,res)
            w, h = x2-x1, y2-y1

            print(res)

            cvzone.cornerRect(img, (x1, y1, w, h),l=9,rt=2,colorR=clr["purple"])
            cvzone.putTextRect(img, f'{track_id}', (max(0, x1), max(35, y1)), scale=0.6, thickness=2,offset=10)
            # cvzone.putTextRect(img, f'{track_id}', (max(0, x1), max(35, y1)), scale=2, thickness=3,offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv.circle(img, (cx, cy), 5,clr["purple"], cv.FILLED)

            if limits[0] < cx < limits[2] and limits[1] - 14 < cy < limits[1] + 14:
                if Count.count(track_id) == 0:
                    Count.append(track_id)
                    cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        cv.putText(img,f"count: {len(Count)}",(50,50),cv.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

        cv.imshow("Image", img)
        k = cv.waitKey(1) & 0xFF
        if k == ord('q'):
            cap.release()
            cv.destroyAllWindows()



  
   
# detect(0)
# detect("VIDEOS/bikes.mp4")
# detect("VIDEOS/people.mp4")
# detect("VIDEOS/motorbikes-1.mp4")
# tracker()
