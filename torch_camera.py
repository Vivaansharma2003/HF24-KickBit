from torch_detection import ReturnLoadedCNN
import cv2
import geocoder
import requests

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

                # Don't actually want to send an SOS to the government since it is testing, so avoiding this part of the code.
                '''g = geocoder.ip('me')
                print(g.latlng)

                api_url = "https://emergency-api.example.in/report"  #Example - Haven't added the actual government emergency API

                latitude = g.latlng[0]  
                longitude = g.latlng[1]

                data = {
                        "message": "SOS! Accident detected",
                        "latitude": latitude,
                        "longitude": longitude
                        }

                # Send POST request with accident data
                response = requests.post(api_url, json=data)

                if response.status_code == 200:
                    print("SOS message sent successfully:", response.text)
                else:
                    print("Error sending SOS message:", response.status_code, response.text)'''

            if cv2.waitKey(33) & 0xFF == ord('q'):
                return
            cv2.imshow('Video', frame)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_application()