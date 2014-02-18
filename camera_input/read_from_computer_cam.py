import cv2


def get_cv_img_from_computer_cam(camera_id):
    camera = cv2.VideoCapture(camera_id)
    while True:
        success, data = camera.read()
        if success:
            yield data
