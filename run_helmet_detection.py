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
col = db['helmets']

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
  helmet_model = YOLO("models/helmet.pt")
  print(helmet_model.names)

  # Model for license plate detection
  plate_model = YOLO("models/license_plate_detector.pt")

  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 1
  color = (255, 0, 0)
  thickness = 2

  while True:
    
    success, img = cap.read()

    if success:
      no_helmet = helmet_model.predict(img, stream=True, conf=0.3)

      # Coordinates
      for res in no_helmet:
        res_boxes = res.boxes
        

        for res_box in res_boxes:
    # Class name
            no_helmet_class_name = "no_helmet"

            plate_results = plate_model(img, stream=True, conf=0.2)

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
                    ocr(img, x1, y1, x2, y2, confidence)

                    # Put box in image
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Object details
                    org = [x1, y1]
                    cv2.putText(img, plate_class_name, org, font, fontScale, color, thickness)

            # Bounding box
            x1, y1, x2, y2 = res_box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Check if the detected object is labeled as 1 (assuming 1 represents number plates)
            if res_box.cls[0] == 1:
                # Put box in frame
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Object details
                org = [x1, y1]
                cv2.putText(img, no_helmet_class_name, org, font, fontScale, color, thickness)


            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break

    else:
        break

  # Release resources
  cap.release()
  cv2.destroyAllWindows()
