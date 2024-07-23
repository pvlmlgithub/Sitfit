from mmpose.apis import MMPoseInferencer
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        print("Faces detected in the image")
    else:
        print("No faces detected in the image")

# Call the function with the path to your image
detect_faces("/home/vinayaka/pvl/flask_app/uploads/car10.png_2024-02-08T11_22_34.452Z_output_2.jpeg")