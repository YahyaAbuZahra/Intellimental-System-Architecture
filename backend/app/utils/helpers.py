
import cv2
import numpy as np

def error_response(message, code="bad_request", http_status=400):
    from flask import jsonify
    return jsonify(
        {
            "ok": False,
            "message": message,
            "error_code": code,
        }
    ), http_status



# FACE EXTRACTION

def extract_face(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    (x, y, w, h) = faces[0]

    face_img = image[y:y+h, x:x+w]
    return face_img


# BRIGHTNESS CALCULATION

def calc_brightness(image):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


# SHARPNESS CALCULATION

def calc_sharpness(image):
  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())
