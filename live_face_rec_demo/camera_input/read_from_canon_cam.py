import cv2
# import numpy as np
from boost_camera import CameraManager
import os
import time

MAX_CONNECTION_RETRIES = 5


def get_prepped_camera_instance():
    print "Instantiating camera"
    camera_manager = CameraManager()
    print "Trying to connect to the camera..."
    while True:
        time.sleep(2)
        print "Killing interfering camera processes..."
        os.popen("killall PTPCamera")
        success = camera_manager.initialize()
        if success:
            print "Successfully connected to the camera"
            break
    return camera_manager


def image_yielder(camera_manager):
    while True:
        numpy_array_jpeg_bytes = camera_manager.grab_frame()
        cv_img = cv2.imdecode(numpy_array_jpeg_bytes, 1)
        if len(numpy_array_jpeg_bytes) > 0:
            yield cv_img


def get_cv_img_from_canon_cam():
    camera_manager = get_prepped_camera_instance()
    for cv_img_array in image_yielder(camera_manager):
        yield cv_img_array


if __name__ == "__main__":
    for cv_img_array in get_cv_img_from_canon_cam():
        pass
