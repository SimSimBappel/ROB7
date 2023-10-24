import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MAX_BUFFER_SIZE = 100  # Maximum number of coordinates to store

MIN_MATCH_COUNT = 10

# Initialize SIFT detector
sift = cv.SIFT_create()

# Load the query image
# img1 = cv.imread('/home/simon/Desktop/ROB7/Vision/testImages/test2.jpg', cv.IMREAD_GRAYSCALE) # queryImage
img1 = cv.imread('/home/simon/Pictures/Webcam/cone.png') # queryImage

kp1, des1 = sift.detectAndCompute(img1, None)

# Initialize video capture (change the argument to 0 if you want to use the webcam, or put the path to a video file)
cap = cv.VideoCapture(0)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Initialize transformation matrix (M)
M = None

fig = plt.subplots()

# Lists to store x and y coordinates over time with a maximum buffer size
x_coords = []
y_coords = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(gray, None)

    if des is not None:

        matches = flann.knnMatch(des1, des, k=2)

        # Store all the good matches as per Lowe's ratio test.
        good = []

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            try:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                frame = cv.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv.LINE_AA)

                # Extract translation vector from the transformation matrix
                translation_vector = M[:, -1]

                # Append x and y coordinates to lists and limit the buffer size
                x_coords.append(translation_vector[0])
                y_coords.append(translation_vector[1])
                if len(x_coords) > MAX_BUFFER_SIZE:
                    x_coords.pop(0)
                    y_coords.pop(0)

            except Exception as e:
                print(f"Exception: {e}")
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        frame = cv.drawMatches(img1, kp1, frame, kp, good, None, **draw_params)

        # Display the frame with matches
        cv.imshow('Frame', frame)

        # Check for user exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        plt.plot(x_coords, label='X Coordinate')
        plt.plot(y_coords, label='Y Coordinate')
        plt.xlabel('Frame Number')
        plt.ylabel('Coordinate Value')
        plt.title('X and Y Coordinates over Time (Live)')
        plt.legend()
        plt.pause(0.01)
        plt.clf()

# Release video capture and destroy windows
cap.release()
cv.destroyAllWindows()
