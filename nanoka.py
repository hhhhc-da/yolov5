# Ultralytics YOLOv5, AGPL-3.0 license
from utils.general import cv2, non_max_suppression
from models.common import DetectMultiBackend
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import threading

# ----------------------------------------------------------------------------------------------------------------------
# Model parameters
data = 'dataset/nanoka/data.yaml'  # dataset.yaml path
imgsz = (640, 640)  # inference size (height, width)
device = "cpu"  # cuda device, i.e. 0 or 0,1,2,3 or cpu

# YOLOv5 model handler
model = None

app = Flask(__name__)
CORS(app)


# ----------------------------------------------------------------------------------------------------------------------
def plot_xyxy(x, img, color=(0, 0, 255), label=None, line_thickness=3):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness,
                  lineType=cv2.LINE_AA)

    if label:
        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(
            label, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)


def inference(img):
    ret = {'type': 'myCar', 'xyxy': [], 'conf': 0.0}

    # NHWC to NCHW
    image = img.copy()
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image) / 255.0

    # Inference
    pred = model(torch.from_numpy(image).unsqueeze(0).float())

    # NMS algorithm
    pred = non_max_suppression(pred)

    # Process predictions
    for i, det in enumerate(pred):
        if len(det):
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                # Trained myCar type
                if c != 80:
                    continue

                if float(conf.item()) > ret['conf']:
                    ret['conf'] = float(conf.item())
                    ret['xyxy'] = [float(xyxy[i].item()) for i in range(4)]

    return ret


def val_vedio(camera):
    cap = cv2.VideoCapture(camera)

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, imgsz)

        # NHWC to NCHW
        image = frame.copy()
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image) / 255.0

        # Inference
        pred = model(torch.from_numpy(image).unsqueeze(0).float())

        # NMS algorithm
        pred = non_max_suppression(pred)

        # Process predictions
        for i, det in enumerate(pred):
            if len(det):
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    # Trained myCar type
                    if c != 80:
                        continue

                    label = '{} {:4f}'.format(
                        model.names[c], float(conf.item()))

                    plot_xyxy(xyxy, frame, label=label,
                              color=(0, 0, 255), line_thickness=2)

                    # print("xyxy:", [float(xyxy[i].item())
                    #       for i in range(4)], "\tconf:", float(conf.item()))

            # Stream results
            cv2.imshow('nanoka_v1', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Done.")
                return


@app.route('/', methods=['GET'])
def root():
    return render_template('50x.html'), 500


@app.route('/val', methods=['GET'])
def test():
    thread1 = threading.Thread(target=val_vedio, args=[1,])
    thread1.start()
    return jsonify({'code': 0, 'info': 'Success'}), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'code': 1, 'info': 'No image file'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    ret = inference(img)
    xyxy = ret['xyxy']
    conf = ret['conf']

    data = [0, 0]

    if conf > 0.3:
        data = [(xyxy[0] + xyxy[2])/2, (xyxy[1] + xyxy[3])/2]
        return jsonify({'code': 0, 'info': 'Success', 'data': str(data)}), 200
    else:
        return jsonify({'code': 2, 'info': 'No myCar'}), 200


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load YOLOv5 model
    model = DetectMultiBackend('nanoka_v1.pt', data=data)

    app.run('0.0.0.0', port=5000)
