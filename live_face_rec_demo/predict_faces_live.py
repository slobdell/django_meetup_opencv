# check out http://www.clipconverter.cc/ to get clips

import cv
import cv2
import datetime
import json
import numpy as np
import os

# from read_from_web_cam import get_cv_img_from_ip_cam
from camera_input.read_from_canon_cam import get_cv_img_from_canon_cam
from utils.face_utils import face_detect_on_photo, normalize_face_for_save, OUTPUT_DIRECTORY, get_recognizer


EXTENSION = ".jpg"
BLUR_THRESHOLD = 6  # higher value means more strict


global_label_dict = {}


def observe_faces(recognizer):
    cv2.namedWindow("Live View", cv2.CV_WINDOW_AUTOSIZE)
    cv2.namedWindow("Face", cv2.CV_WINDOW_AUTOSIZE)
    print "starting video capture..."
    while True:
        for cv_array in get_cv_img_from_canon_cam():
            try:
                cv2.imshow("Live View", cv_array)  # cv_array is a numpy array
                cv2.waitKey(1)
                img = cv.fromarray(cv_array)
            except (TypeError, cv2.error):
                print "Warning...got malformed JPEG data"
                continue  # cv_array was malformed, ignore and move to next frame
            detect_and_handle_faces(img, recognizer)


def face_is_blurry(face):
    im = cv2.cv.fromarray(face)
    col_edge = cv.CreateImage((im.width, im.height), 8, 1)

    # convert to grayscale
    # gray = cv.CreateImage((im.width, im.height), 8, 1)
    gray = im
    edge = cv.CreateImage((im.width, im.height), 8, 1)
    # cv.CvtColor(im, gray, cv.CV_BGR2GRAY)

    cv.Smooth(gray, edge, cv.CV_BLUR, 3, 3, 0)
    cv.Not(gray, edge)
    position = 150
    cv.Canny(gray, edge, position, position * 3, 3)
    cv.SetZero(col_edge)
    cv.Copy(im, col_edge, edge)
    avg = np.average(np.asarray(edge[:, :]))
    if avg > BLUR_THRESHOLD:
        return False
    return True


def detect_and_handle_faces(img, recognizer=None):
    faces = face_detect_on_photo(img)
    for face in faces:
        face = normalize_face_for_save(face)
        face = np.asarray(face)
        if face_is_blurry(face):
            continue
        cv2.imshow("Face", face)
        if recognizer is not None:
            [label_id, confidence] = recognizer.predict(face)
            person = get_person_from_label(label_id)
            print "Predicting %s with %s confidence" % (person, confidence)
        else:
            label_id = None
        save_face(face, label_id)


def save_face(face, label_id):
    person = get_person_from_label(label_id) if label_id is not None else ""
    filename = datetime.datetime.now().strftime("%m%d%Y_%H%M%S_%f")
    canonical_person = person.lower().replace(" ", "_")
    filename = "_" + filename + "_" + canonical_person
    full_path = "/".join((OUTPUT_DIRECTORY, filename,)) + EXTENSION
    cv2.imwrite(full_path, face)


def get_person_from_label(label_id):
    key = "global_label_key"
    label_id = str(label_id)
    if key in global_label_dict:
        label_dict = global_label_dict[key]
    else:
        with open("labels.txt", "r") as file:
            json_str = file.read()
        label_dict = json.loads(json_str)
        global_label_dict[key] = label_dict
    return label_dict[label_id]


def recognize_faces():
    recognizer = get_recognizer()
    observe_faces(recognizer)


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    recognize_faces()
