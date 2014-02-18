import cv
import cv2
import numpy as np


def no_action(frame):
    return frame


def edge_detect(frame):
    im = cv2.cv.fromarray(frame)
    col_edge = cv.CreateImage((im.width, im.height), 8, 3)

    # convert to grayscale
    gray = cv.CreateImage((im.width, im.height), 8, 1)
    cv.CvtColor(im, gray, cv.CV_BGR2GRAY)
    edge = cv.CreateImage((im.width, im.height), 8, 1)

    cv.Smooth(gray, edge, cv.CV_BLUR, 3, 3, 0)
    cv.Not(gray, edge)
    threshold = 30
    cv.Canny(gray, edge, threshold, threshold * 3, 3)
    cv.SetZero(col_edge)
    cv.Copy(im, col_edge, edge)
    return np.asarray(edge[:, :])
    return np.asarray(col_edge[:, :])


def grayscale(frame):
    im = cv2.cv.fromarray(frame)
    gray = cv.CreateImage((im.width, im.height), 8, 1)
    cv.CvtColor(im, gray, cv.CV_BGR2GRAY)
    return np.asarray(gray[:, :])


CASCADE = cv.Load("./text_files/haarcascade_frontalface_alt.xml")
SMILE_CASCADE = cv.Load("./text_files/haarcascade_smile.xml")
min_size = (20, 20)
image_scale = 2
haar_scale = 1.1
min_neighbors = 2
haar_flags = 0


def face_detect(frame):
    img = cv.fromarray(frame)
    # allocate temporary images
    gray = cv.CreateImage((img.width, img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
                   cv.Round(img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)

    faces = cv.HaarDetectObjects(small_img, CASCADE, cv.CreateMemStorage(0),
                                 haar_scale, min_neighbors, haar_flags, min_size)
    if faces:
        for ((x, y, w, h), n) in faces:
            # the input to cv.HaarDetectObjects was resized, so scale the
            # bounding box of each face and convert it to two CvPoints
            pt1 = (int(x * image_scale), int(y * image_scale))
            pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
            cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)
    return np.asarray(img)


def smile_detect(frame):
    img = cv.fromarray(frame)
    # allocate temporary images
    gray = cv.CreateImage((img.width, img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
                   cv.Round(img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)

    faces = cv.HaarDetectObjects(small_img, CASCADE, cv.CreateMemStorage(0),
                                 haar_scale, min_neighbors, haar_flags, min_size)
    min_smile_size = (5, 5)
    if faces:
        for ((x, y, w, h), n) in faces:
            # pt1 = (int(x * image_scale), int(y * image_scale))
            # pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            # face = img[pt1[1]:pt2[1], pt1[0]: pt2[0]]
            face = small_img[pt1[1]:pt2[1], pt1[0]: pt2[0]]
            smiles = cv.HaarDetectObjects(face, SMILE_CASCADE, cv.CreateMemStorage(0),
                                          haar_scale, min_neighbors, haar_flags, min_smile_size)
            for ((x2, y2, w2, h2), n2) in smiles:
                pt1 = (int((x + x2) * image_scale), int((y + y2) * image_scale))
                pt2 = (int((x + x2 + w2) * image_scale), int((y + y2 + h2) * image_scale))
                cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)
    return np.asarray(img)


def laplace(frame):
    colorlaplace = None
    planes = [None, None, None]
    frame = cv.fromarray(frame)
    laplace_var = None

    planes = [cv.CreateImage((frame.width, frame.height), 8, 1) for i in range(3)]
    laplace_var = cv.CreateImage((frame.width, frame.height), cv.IPL_DEPTH_16S, 1)
    colorlaplace = cv.CreateImage((frame.width, frame.height), 8, 3)

    cv.Split(frame, planes[0], planes[1], planes[2], None)
    for plane in planes:
        cv.Laplace(plane, laplace_var, 3)
        cv.ConvertScaleAbs(laplace_var, plane, 1, 0)

    cv.Merge(planes[0], planes[1], planes[2], None, colorlaplace)
    return np.asarray(colorlaplace[:, :])

# this is super hacky for demonstration purposes.  Do NOT do this in any
# sort of production environment.  This only works because my camera
# sizes are of different length
previous_frame_manager = {}


def motion_detect(frame):
    previous_frame = previous_frame_manager.get(len(frame[0]), (None, None,))[0]
    previous_previous_frame = previous_frame_manager.get(len(frame[0]), (None, None,))[1]
    return_frame = None
    if previous_previous_frame is not None:
        d1 = cv2.absdiff(frame, previous_frame)
        d2 = cv2.absdiff(previous_frame, previous_previous_frame)
        return_frame = cv2.bitwise_xor(d1, d2)
    previous_previous_frame = previous_frame
    previous_frame = frame
    previous_frame_manager[len(frame[0])] = (previous_frame, previous_previous_frame, )
    return return_frame


def color_detect(frame):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # color has been optimized to detect the blue recycle bins in the office
    blue_min = np.array([90, 90, 90], np.uint8)
    blue_max = np.array([150, 250, 250], np.uint8)
    threshold = cv2.inRange(np.asarray(hsv_img),
                            blue_min,
                            blue_max)
    return threshold
