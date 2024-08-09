import requests
import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(1)
ret, frame = cap.read()
frame = cv2.resize(frame, (640, 640))

_, img_encoded = cv2.imencode('.jpg', frame)

response = requests.post(
    'http://127.0.0.1:5000/analyze',
    files={'image': img_encoded.tobytes()}
)


print(response)
json_data = response.json()
print(json_data)

if (json_data['code'] == 0):
    point = [int(float(istr)) for istr in json_data['data'][1:-1].split(',')]

    cv2.circle(frame, point, radius=5,
               color=(0, 0, 255), thickness=10)

cap.release()

plt.figure()
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.show()
