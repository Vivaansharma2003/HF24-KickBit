from torch_detection import ReturnLoadedCNN
import cv2

model = ReturnLoadedCNN(path_to_weights='models/crash_cnn.pth')
font = cv2.FONT_HERSHEY_SIMPLEX

def start_application():
    cap = cv2.VideoCapture(0) # for camera use cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret == True:
            pred, prob = model.predict_accident(frame)
            if(pred == "Accident"):
                prob = (round(prob.item()*100, 2))

                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                return
            cv2.imshow('Video', frame)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_application()