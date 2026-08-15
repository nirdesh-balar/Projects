import bz2
import os
import urllib.request

import dlib
import numpy as np
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../models")
SHAPE_MODEL = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
FACE_MODEL = os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")

SHAPE_MODEL_URL = (
    "https://github.com/davisking/dlib-models/raw/master/"
    "shape_predictor_68_face_landmarks.dat.bz2"
)
FACE_MODEL_URL = (
    "https://github.com/davisking/dlib-models/raw/master/"
    "dlib_face_recognition_resnet_model_v1.dat.bz2"
)


def _download_and_extract(url, output_path):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    compressed_path = output_path + ".bz2"
    try:
        urllib.request.urlretrieve(url, compressed_path)
        with bz2.open(compressed_path, "rb") as source, open(output_path, "wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
    finally:
        if os.path.exists(compressed_path):
            os.remove(compressed_path)

    return output_path


@st.cache_resource(show_spinner="Loading face recognition models...")
def load_dlib_models():
    """Load dlib models without installing face_recognition_models.

    The model files are downloaded only once per Streamlit app instance.
    This avoids the large face-recognition-models pip package that was
    timing out during Streamlit Cloud dependency installation.
    """
    try:
        shape_path = _download_and_extract(SHAPE_MODEL_URL, SHAPE_MODEL)
        face_path = _download_and_extract(FACE_MODEL_URL, FACE_MODEL)

        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(shape_path)
        facerec = dlib.face_recognition_model_v1(face_path)
        return detector, sp, facerec
    except Exception as exc:
        st.error(f"Unable to load face recognition models: {exc}")
        raise


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)

    encodings = []
    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
        encodings.append(np.array(face_descriptor))

    return encodings


@st.cache_resource
def get_trained_model():
    X = []
    y = []

    student_db = get_all_students()
    if not student_db:
        return None

    for student in student_db:
        embedding = student.get("face_embedding")
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get("student_id"))

    if len(X) == 0:
        return None

    clf = SVC(kernel="linear", probability=True, class_weight="balanced")

    # SVC requires at least two classes. For a single enrolled student,
    # recognition is handled directly in predict_attendance().
    if len(set(y)) >= 2:
        clf.fit(X, y)
        clf = clf
    else:
        clf = None

    return {"clf": clf, "X": X, "y": y}


def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)


def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)
    detected_student = {}

    model_data = get_trained_model()
    if not model_data:
        return detected_student, [], len(encodings)

    clf = model_data["clf"]
    X_train = model_data["X"]
    y_train = model_data["y"]
    all_students = sorted(list(set(y_train)))

    for encoding in encodings:
        if clf is not None:
            predicted_id = int(clf.predict([encoding])[0])
        else:
            predicted_id = int(all_students[0])

        # Compare against all stored embeddings for the predicted student.
        student_embeddings = [
            X_train[i] for i, sid in enumerate(y_train) if sid == predicted_id
        ]
        best_match_score = min(
            np.linalg.norm(embedding - encoding)
            for embedding in student_embeddings
        )

        resemblance_threshold = 0.6
        if best_match_score <= resemblance_threshold:
            detected_student[predicted_id] = True

    return detected_student, all_students, len(encodings)
