from flask import Flask, request, jsonify
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from PIL import Image
import pandas as pd
from mmpose.apis import MMPoseInferencer
inferencer = MMPoseInferencer(pose2d='human', pose2d_weights='/home/vinayaka/Downloads/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth', det_model='whole_image')
import cv2
from rmn import RMN
m = RMN()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        return "face"
    else:
        return "no_face"

def stress(result):
    emo_label = result[0]['emo_label']
    if emo_label in ["happy", "surprise", "neutral"]:
        output = "no_stress"
    else:
        output = "stress"
    return output

app = Flask(__name__)
columns = ['nose_x', 'nose_y', 'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y', 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y']

# Directory to store uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/stress', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    model_file = open('/home/vinayaka/pvl/flask_app/model.pkl', 'rb')
    model = pickle.load(model_file)
    model_file.close()
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    upload_file():
    file = request.files.get('file')
    model_file.close() file.close()model_file.close()    model_file.close()
    model_file = open('/home/vinayaka/pvl/flask_app/model.pkl', 'rb')
    model = pickle.load(model_file)
    model_file.close()
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    result_generator = inferencer(filename, out_dir = "/home/vinayaka/pvl/flask_app/output", return_vis=True, radius=12, thickness=5, draw_bbox=False, draw_heatmap=False)
    result = next(result_generator)
    width = result["visualization"][0].shape[1]
    row = []
    keypoints = result['predictions'][0][0]["keypoints"]
    keypoints = keypoints[:7]
    result_generator = inferencer(filename, out_dir = "/home/vinayaka/pvl/flask_app/output", return_vis=True, radius=12, thickness=5, draw_bbox=False, draw_heatmap=False)
    result = next(result_generator)
    width = result["visualization"][0].shape[1]
    row = []
    keypoints = result['predictions'][0][0]["keypoints"]
    keypoints = keypoints[:7]

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    model_file = open('/home/vinayaka/pvl/flask_app/model.pkl', 'rb')
    model = pickle.load(model_file)
    model_file.close()

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    result_generator = inferencer(filename, out_dir = "/home/vinayaka/pvl/flask_app/output", return_vis=True, radius=12, thickness=5, draw_bbox=False, draw_heatmap=False)
    result = next(result_generator)
    width = result["visualization"][0].shape[1]
    row = []
    keypoints = result['predictions'][0][0]["keypoints"]
    keypoints = keypoints[:7]
    for keypoint_pair in keypoints:
        for index,keypoint in enumerate(keypoint_pair):
            if index==0:
                row.append(width-keypoint)
            else:
                row.append(keypoint)
    X_inference = pd.DataFrame([row], columns=columns)
    print(X_inference)
    y_pred = model.predict(X_inference)
    y_pred = y_pred[0]
    if y_pred==0:
        posture_type = "improper"
        print("IMPROPER POSTURE")
    elif y_pred==1:
        print("PROPER POSTURE")
        posture_type = "proper"
    return jsonify({'Posture': posture_type, 'filename': filename})

if __name__ == '__main__':
    app.run(debug=True)