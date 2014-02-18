import cv2
import datetime
import json
import numpy as np
import os
import random

from collections import defaultdict

# IMPORTANT:  This is set to a low number to get your training data
# initialized, but you'll want to raise this number as you get a bigger
# training set.
FACES_PER_CLASS = 10

CASCADE = "./text_files/haarcascade_frontalface_alt.xml"
cascade = cv2.cv.Load(CASCADE)
OUTPUT_DIRECTORY = "./faces"
haar_scale = 1.2  # initially set to 1.2
IMAGE_SCALE = 2
min_neighbors = 4  # good was set to 3
min_size = (20, 20)
haar_flags = 0
normalized_face_dimensions = (100, 100)
RECOGNIZER_FILENAME = "./text_files/saved_recognizer.xml"
global_recognizer = {}


def face_detect_on_photo(img):
    faces = []
    small_img = normalize_image_for_face_detection(img)
    faces_coords = cv2.cv.HaarDetectObjects(small_img, cascade, cv2.cv.CreateMemStorage(0),
                                        haar_scale, min_neighbors, haar_flags, min_size)
    for ((x, y, w, h), n) in faces_coords:
        pt1 = (int(x * IMAGE_SCALE), int(y * IMAGE_SCALE))
        pt2 = (int((x + w) * IMAGE_SCALE), int((y + h) * IMAGE_SCALE))
        face = img[pt1[1]:pt2[1], pt1[0]: pt2[0]]
        face = normalize_face_for_save(face)
        faces.append(face)
    return faces


def normalize_image_for_face_detection(img):
    gray = cv2.cv.CreateImage((img.width, img.height), 8, 1)
    small_img = cv2.cv.CreateImage((cv2.cv.Round(img.width / IMAGE_SCALE),
                   cv2.cv.Round(img.height / IMAGE_SCALE)), 8, 1)
    if img.channels > 1:
        cv2.cv.CvtColor(img, gray, cv2.cv.CV_BGR2GRAY)
    else:
        # image is already grayscale
        gray = cv2.cv.CloneMat(img[:, :])
    cv2.cv.Resize(gray, small_img, cv2.cv.CV_INTER_LINEAR)
    cv2.cv.EqualizeHist(small_img, small_img)
    # uncomment for debugging
    # cv2.imshow("img", np.asarray(small_img[:, :]))
    return small_img


def normalize_face_for_save(face):
    face = normalize_face_size(face)
    face = normalize_face_color(face)
    face = normalize_face_histogram(face)
    return face


def normalize_face_size(face):
    normalized_face_dimensions = (100, 100)
    face_as_array = np.asarray(face)
    resized_face = cv2.resize(face_as_array, normalized_face_dimensions)
    resized_face = cv2.cv.fromarray(resized_face)
    return resized_face


def normalize_face_histogram(face):
    face_as_array = np.asarray(face)
    equalized_face = cv2.equalizeHist(face_as_array)
    equalized_face = cv2.cv.fromarray(equalized_face)
    return equalized_face


def normalize_face_color(face):
    gray_face = cv2.cv.CreateImage((face.width, face.height), 8, 1)
    if face.channels > 1:
        cv2.cv.CvtColor(face, gray_face, cv2.cv.CV_BGR2GRAY)
    else:
        # image is already grayscale
        gray_face = cv2.cv.CloneMat(face[:, :])
    return gray_face[:, :]


def prune_labels_and_images(labels, images, label_dict):
    min_num_faces = FACES_PER_CLASS
    label_count = defaultdict(int)

    for label_id in labels:
        label_count[label_id] += 1

    label_count = {k: v for k, v in label_count.items() if v >= min_num_faces}

    max_num_faces = min(label_count.values())
    for label_id in label_count.keys():
        if label_count[label_id] == max_num_faces:
            limiting_id = label_id

    print "Pruning data so that all classes have %s faces" % max_num_faces
    print "Class with the lowest number of faces is %s" % label_dict[limiting_id]

    new_images = []
    new_labels = []
    current_count = defaultdict(int)
    for index in xrange(len(labels)):
        label_id = labels[index]
        image = images[index]
        current_count[label_id] += 1
        if current_count[label_id] <= max_num_faces and label_id in label_count:
            new_images.append(image)
            new_labels.append(label_id)
    images = new_images
    labels = new_labels
    label_dict = {k: v for k, v in label_dict.items() if k in labels}
    print "%s total people are in the training set" % len(label_dict)
    label_json = json.dumps(label_dict)
    with open("labels.txt", "w+") as file:
        file.write(label_json)
    print label_dict
    return labels, images


def get_sharpness_from_filename(full_filename):
    im = cv2.imread(full_filename)
    if im is None:
        print "IM IS NONE"
        return 0
    im = cv2.cv.fromarray(im)
    # create the output im
    col_edge = cv2.cv.CreateImage((im.width, im.height), 8, 3)

    # convert to grayscale
    gray = cv2.cv.CreateImage((im.width, im.height), 8, 1)
    edge = cv2.cv.CreateImage((im.width, im.height), 8, 1)
    cv2.cv.CvtColor(im, gray, cv2.cv.CV_BGR2GRAY)

    cv2.cv.Smooth(gray, edge, cv2.cv.CV_BLUR, 3, 3, 0)
    cv2.cv.Not(gray, edge)
    position = 150
    cv2.cv.Canny(gray, edge, position, position * 3, 3)
    cv2.cv.SetZero(col_edge)
    cv2.cv.Copy(im, col_edge, edge)
    avg = np.average(np.asarray(col_edge[:, :]))
    return avg


def order_filenames_by_blurriness(image_list, base_directory):
    list_full_paths = [base_directory + image_name for image_name in image_list]
    list_full_paths = sorted(list_full_paths, key=get_sharpness_from_filename, reverse=True)
    image_list = [path.replace(base_directory, "") for path in list_full_paths]
    return image_list


def train_recognizer(recognizer):
    images = []
    labels = []
    label_dict = {}  # dictionary of label ID's to a person's name
    all_people = os.listdir(OUTPUT_DIRECTORY)
    all_people = [item for item in all_people if "DS_Store" not in item]
    id_counter = 1
    for person in all_people:
        directory = "/".join((OUTPUT_DIRECTORY, person,))
        directory += "/"
        try:
            all_pictures = os.listdir(directory)
        except OSError:
            continue
        all_pictures = [item for item in all_pictures if "DS_Store" not in item]
        random.shuffle(all_pictures)  # shuffle since image list will later be truncated
        # all_pictures = order_filenames_by_blurriness(all_pictures, directory)[:FACES_PER_CLASS]  # TODO: verify whether or not this is needed
        for picture_name in all_pictures:
            full_path = directory + picture_name
            try:
                face = cv2.cv.LoadImage(full_path, cv2.IMREAD_GRAYSCALE)
            except IOError:
                print "WARNING!  Could not open %s" % full_path
                continue
            images.append(np.asarray(face[:, :]))
            labels.append(id_counter)
        label_dict[id_counter] = person
        id_counter += 1
    labels, images = prune_labels_and_images(labels, images, label_dict)
    image_array = np.asarray(images)
    label_array = np.asarray(labels)
    print "Starting the training of the recognizer.  This may take a bit..."
    start = datetime.datetime.now()
    recognizer.train(image_array, label_array)
    end = datetime.datetime.now()
    num_people = len(set(labels))
    print "Finished training with %s people in %s minutes" % (num_people, round((end - start).total_seconds() / 60, 2))
    return recognizer


def get_recognizer():
    key = "global_recognizer_key"
    if key in global_recognizer:
        return global_recognizer[key]
    recognizer = cv2.createFisherFaceRecognizer()
    try:
        recognizer.load(RECOGNIZER_FILENAME)
    except cv2.error:
        print "Starting training...",
        recognizer = train_recognizer(recognizer)
        print "finished"
        global_recognizer[key] = recognizer
        recognizer.save(RECOGNIZER_FILENAME)
    return recognizer
