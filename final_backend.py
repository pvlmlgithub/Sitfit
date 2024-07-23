"""
randomized search cv
grid search cv
apply l1 l2 regulaization, adam optimizer, 1024 1024 512 256 layers
reduce the loss from 5 digits to 1 digit
Loss must be less than 1
"""

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
# inferencer = MMPoseInferencer(pose2d='human', pose2d_weights='/home/vinayaka/Downloads/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth', det_model='whole_image')
import cv2
app = Flask(__name__)
columns = ['nose_x', 'nose_y', 'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y', 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
from ResidualMaskingNetwork.rmn import RMN
import cv2
m = RMN()

def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        return True
    else:
        return False

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Flask app to store images in the specified directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    model_file = open('/home/vinayaka/pvl/flask_app/model.pkl', 'rb')
    model = pickle.load(model_file)
    model_file.close()

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    if not detect_faces(filename):
        return jsonify({"Error": "Input is not a persons image"})

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
        filename = os.path.join("improper", file.filename)
        file.save(filename)

    elif y_pred==1:
        print("PROPER POSTURE")
        posture_type = "proper"
        filename = os.path.join("proper", file.filename)
        file.save(filename)
    return jsonify({'Posture': posture_type, 'filename': filename})

@app.route('/detect_stress', methods=['POST'])
def detect_stress():    
    file = request.files.get('file')
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    image = cv2.imread(filename)
    results = m.detect_emotion_for_single_frame(image)
    results = results[0]
    stress_emotions = ['angry', 'sad', 'fear', 'disgust']
    stress_threshold = 0.5  # Define a threshold for stress classification
    print(results)
    total_stress_proba = 0.0
    for emotion_dict in results["proba_list"]:
        emotion_tuple = list(emotion_dict.items())[0]
        emotion, prob = emotion_tuple
        if emotion in stress_emotions:
            total_stress_proba += prob
    is_stressed = total_stress_proba >= stress_threshold
    stress_result = "Stress" if is_stressed else "No Stress"
    return jsonify({'Result': stress_result})

if __name__ == '__main__':
    app.run(debug=True)