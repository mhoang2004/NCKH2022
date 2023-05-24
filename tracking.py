import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
from termcolor2 import colored

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']


def draw_boundingBox(frame, bb_keypoints, confidence_threshold, cur_center_points, pre_center_points):
    y, x, c = frame.shape
    bb_shaped = np.squeeze(np.multiply(bb_keypoints, [y, x, y, x, 1]))
    for kb in bb_shaped:
        ymin, xmin, ymax, xmax, score = kb

#         print(int(ymin), int(xmin), int(ymax), int(xmax), score)

        if score > confidence_threshold:
            cx = int((xmin + xmax)/2)
            cy = int((ymin + ymax)/2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.rectangle(frame, (int(xmin), int(ymax)),
                          (int(xmax), int(ymin)), (0, 255, 0), 2)
#             print(cx, cy)

            if (cx, cy) not in cur_center_points:
                cur_center_points.append((cx, cy))


def loop_through_people(frame, confidence_threshold, bb_keypoints_with_scores, cur_center_points, pre_center_points, count, rect):
    for bb_person in bb_keypoints_with_scores:
        draw_boundingBox(frame, bb_keypoints_with_scores,
                         confidence_threshold, cur_center_points, pre_center_points)

        # Check if any person is inside the rectangle
        for pt in cur_center_points:
            if rect[0][0] <= pt[0] <= rect[1][0] and rect[0][1] <= pt[1] <= rect[1][1]:
                print(colored("Person inside the rectangle: True", "green"))
                break
        else:
            print(colored("Person inside the rectangle: False", "red"))
            continue
        break

    # Draw rectangle
    cv2.rectangle(frame, rect[0], rect[1], (255, 0, 0), 2)


pre_center_points = []
tracking_objects = {}
track_id = 0
count = 0
k = True
cap = cv2.VideoCapture('7e2.mp4')
while cap.isOpened():
    ret, frame = cap.read()

    count += 1

    if not ret:
        break
    cur_center_points = []
    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 352, 640)
    input_img = tf.cast(img, dtype=tf.int32)

    # Detection section
    results = movenet(input_img)
    bb_keypoints_with_scores = results['output_0'].numpy()[
        :, :, 51:].reshape((6, 5))

    # Render
    rectangle_coordinates = ((400, 100), (600, 400))
    loop_through_people(frame, 0.2, bb_keypoints_with_scores,
                        cur_center_points, pre_center_points, count, rectangle_coordinates)

#     start tracking

    if count <= 2:
        for pt in cur_center_points:
            for pt2 in pre_center_points:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
#                 print("Distance",distance)

                if distance < 50:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        cur_center_points_copy = cur_center_points.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in cur_center_points_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
#                 print("Distance",distance)
                if distance < 50:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in cur_center_points:
                        cur_center_points.remove(pt)
                    continue

            if not object_exists:
                tracking_objects.pop(object_id)

        for pt in cur_center_points:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id),
                    (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

#     print(tracking_objects)

#     end tracking

#     print("Cur Frame", count, cur_center_points)
#     print("Pre Frame", count-1, pre_center_points)
#     print("-----------------------------------")
    cv2.imshow("Multipose", frame)

    while k:
        key = cv2.waitKey(0)
        if key == 27:
            k = False

    pre_center_points = cur_center_points.copy()

#     key = cv2.waitKey(0)
#     if key == 27:
#         break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
