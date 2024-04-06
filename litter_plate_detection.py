from ultralytics import YOLO
from pymongo import MongoClient
from mongopass import mongopass
import numpy as np
from PIL import Image
import pytesseract
import cv2
import math
import io

client = MongoClient(mongopass)
db = client['crud']
col = db['licence_plates']

client = MongoClient()
db = client.testdb
images = db.images

def ocr(frame,x1,y1,x2,y2,score):
  # crop license plate
  license_plate_crop = frame[y1 : y2, x1 : x2, :]

  # process license plate
  license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
  _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

  # read license plate number
  text = pytesseract.image_to_string(license_plate_crop_thresh)

  text = text.upper().replace(' ', '')

  PIL_image = Image.fromarray(np.uint8(license_plate_crop_thresh)).convert('RGB')
  image_bytes = io.BytesIO()

  if text is not None:
    results = {'image': image_bytes.getvalue(),
               'license_plate': {'bbox': [x1, y1, x2, y2],
                                 'text': text,
                                 'bbox_score': score}}
    
    col.insert_one(results)
                    
def video_detection(path_x):
  # Start webcam
  cap = cv2.VideoCapture(path_x)
  cap.set(3, 640)  # Set width
  cap.set(4, 480)  # Set height

  # Model for litter detection
  litter_model = YOLO("models/litter_detection.pt")

  # Model for license plate detection
  plate_model = YOLO("models/license_plate_detector.pt")

  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 1
  color = (255, 0, 0)
  thickness = 2

  while True:
    
    success, img = cap.read()

    if success:
      litter_results = litter_model(img, stream=True, conf=0.3)

      # Coordinates
      for litter in litter_results:
        litter_boxes = litter.boxes

        for litter_box in litter_boxes:
          # Class name
          litter_class_name = litter_model.names[int(litter_box.cls[0])]

          plate_results = plate_model(img, stream=True, conf=0.3)

          # Coordinates
          for plate in plate_results:
              plate_boxes = plate.boxes

              for plate_box in plate_boxes:
                  # Class name
                  plate_class_name = plate_model.names[int(plate_box.cls[0])]

                  # Bounding box
                  x1, y1, x2, y2 = plate_box.xyxy[0]
                  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                  # Confidence
                  confidence = math.ceil((plate_box.conf[0] * 100)) / 100

                  # OCR of the plate
                  ocr(img,x1,y1,x2,y2,confidence)

                  # Put box in image
                  cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                  # Object details
                  org = [x1, y1]
                  cv2.putText(img, plate_class_name, org, font, fontScale, color, thickness)
 
          # Bounding box
          x1, y1, x2, y2 = litter_box.xyxy[0]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

          # Put box in frame
          cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

          # Confidence
          #confidence = math.ceil((litter_box.conf[0] * 100)) / 100

          # Object details
          org = [x1, y1]
          cv2.putText(img, litter_class_name, org, font, fontScale, color, thickness)

      cv2.imshow('Webcam', img)
      if cv2.waitKey(1) == ord('q'):
        break

    else:
      break

  # Release resources
  cap.release()
  cv2.destroyAllWindows()
