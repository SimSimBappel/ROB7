# Code adapted from https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/video/optical_flow/optical_flow.py

import numpy as np
import cv2

cap = cv2.VideoCapture("lec3/exercise_materials/slow_traffic_small.mp4")
cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, current_frame = cap.read()
current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
current_points = cv2.goodFeaturesToTrack(current_frame_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(current_frame)

while True:
    ret, next_frame = cap.read()
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    next_points, points_found, _ = cv2.calcOpticalFlowPyrLK(current_frame_gray, next_frame_gray, current_points, None, **lk_params)

    # Select good points
    if next_points is not None:
        good_points_new = next_points[points_found == 1]
        good_points_current = current_points[points_found == 1]

    # draw the tracks
    for i, (new_point, current_point) in enumerate(zip(good_points_new, good_points_current)):
        a, b = new_point.ravel()
        c, d = current_point.ravel()
        mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        next_frame = cv2.circle(next_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(next_frame, mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 113:
        break

    # Now update the previous frame and previous points
    current_frame_gray = next_frame_gray.copy()
    current_points = good_points_new.reshape(-1, 1, 2)