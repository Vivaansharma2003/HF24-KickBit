from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    # start webcam
    cap = cv2.VideoCapture(path_x)
    cap.set(3, 640)
    cap.set(4, 480)

    out = cv2.VideoWriter('yolov8_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30,(640,480))
    
    # model
    model = YOLO("models/license_plate_detector.pt")
        
    while True:
        success, img = cap.read()

        if success:
          results = model(img, stream=True, conf=0.3)

          # coordinates
          for r in results:
              boxes = r.boxes

              for box in boxes:
                  # bounding box
                  x1, y1, x2, y2 = box.xyxy[0]
                  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                  # put box in cam
                  cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                  # confidence
                  confidence = math.ceil((box.conf[0]*100))/100
                  print("Confidence --->",confidence)

                  # class name
                  cls = int(box.cls[0])
                  print("Class name -->", model.names[cls])

                  # object details
                  org = [x1, y1]
                  font = cv2.FONT_HERSHEY_SIMPLEX
                  fontScale = 1
                  color = (255, 0, 0)
                  thickness = 2

                  cv2.putText(img, model.names[cls], org, font, fontScale, color, thickness)

          out.write(img)

          cv2.imshow('Webcam', img)
          if cv2.waitKey(1) == ord('q'):
            break

        else:
           break

        #yield img


    cap.release()
    cv2.destroyAllWindows()
