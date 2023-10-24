import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

# Initialize SIFT detector
sift = cv.SIFT_create()

# Initialize video capture (change the argument to 0 if you want to use the webcam, or put the path to a video file)
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors with SIFTA
    kp, des = sift.detectAndCompute(gray, None)

    if des is not None:
        img1 = cv.imread('testImages/gel_bottle.jpg', cv.IMREAD_GRAYSCALE) # queryImage
        kp1, des1 = sift.detectAndCompute(img1,None)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []

        for m,n in matches:
            if m.distance < 0.7*n.distance: 
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            frame = cv.polylines(frame,[np.int32(dst)],True,(0,255,0),3, cv.LINE_AA)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            singlePointColor = None,
            matchesMask = matchesMask, # draw only inliers
            flags = 2)

        frame = cv.drawMatches(img1,kp1,frame,kp,good,None,**draw_params)

        # Display the frame with matches
        cv.imshow('Frame', frame)

    # Check for user exit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv.destroyAllWindows()
