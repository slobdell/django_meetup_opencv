# check out http://www.clipconverter.cc/ to get clips

import cv2
# import numpy as np

from camera_input.read_from_web_cam import get_cv_img_from_ip_cam

from sample_functions import color_detect
from sample_functions import edge_detect
from sample_functions import face_detect
from sample_functions import grayscale
from sample_functions import laplace
from sample_functions import no_action
from sample_functions import motion_detect
from sample_functions import smile_detect

TARGET_HEIGHT = 400

WINDOW1 = "IP Camera"
WINDOW2 = "Canon Camera"
WINDOW3 = "Computer Camera"

KEY_TO_FUNCTION = {
        'q': edge_detect,
        'w': laplace,
        'e': grayscale,
        'r': motion_detect,
        't': color_detect,
        'y': face_detect,
        'u': smile_detect,
}
last_key_pressed = 0


def setup():
    cv2.namedWindow(WINDOW1, cv2.CV_WINDOW_AUTOSIZE)
    cv2.namedWindow(WINDOW2, cv2.CV_WINDOW_AUTOSIZE)
    cv2.namedWindow(WINDOW3, cv2.CV_WINDOW_AUTOSIZE)


def _handle_frame(cv_array, window):
    global last_key_pressed
    opencv_func = KEY_TO_FUNCTION.get(chr(last_key_pressed), no_action)
    cv_array = get_resized_image(cv_array)
    cv_array = opencv_func(cv_array)
    if cv_array is not None:
        cv2.imshow(window, cv_array)
        key_pressed = cv2.waitKey(1)
        if key_pressed != -1:
            last_key_pressed = key_pressed
            print "Switching function to %s" % KEY_TO_FUNCTION.get(chr(last_key_pressed), edge_detect).__name__
            if chr(last_key_pressed) == 'x':
                exit()


def web_cam_thread():
    for cv_array in get_cv_img_from_ip_cam():
        _handle_frame(cv_array, WINDOW1)


def get_resized_image(frame):
    height = len(frame)
    width = len(frame[0])
    new_width = TARGET_HEIGHT * width / height
    normalized_dimensions = (new_width, TARGET_HEIGHT)
    return cv2.resize(frame, normalized_dimensions)


if __name__ == "__main__":
    setup()
    web_cam_thread()
