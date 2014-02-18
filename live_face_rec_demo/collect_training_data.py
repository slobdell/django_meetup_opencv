import cv2
import cv
import os
# from read_from_web_cam import get_cv_img_from_ip_cam
from camera_input.read_from_canon_cam import get_cv_img_from_canon_cam
from predict_faces_live import detect_and_handle_faces
from utils.face_utils import OUTPUT_DIRECTORY


def save_faces():
    cv2.namedWindow("Face", cv2.CV_WINDOW_AUTOSIZE)
    while True:
        for cv_array in get_cv_img_from_canon_cam():
            try:
                img = cv.fromarray(cv_array)
            except TypeError:
                print "Warning...got malformed JPEG data"
                continue
            cv2.imshow("Face", cv_array)  # cv_array is a numpy array
            cv2.waitKey(1)  # clear the buffer
            detect_and_handle_faces(img)


if __name__ == "__main__":
    os.nice(19)
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    save_faces()
