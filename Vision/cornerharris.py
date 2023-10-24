import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images
image1 = cv2.imread('testImages/aau-city-1.jpg')
image2 = cv2.imread('testImages/aau-city-2.jpg')

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect Harris corners in both images
def detect_harris_corners(image):
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Find coordinates of Harris corners
    coordinates = np.argwhere(dst > 0.1 * dst.max())

    # Create a list of KeyPoint objects from the Harris corner coordinates
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in coordinates]

    return keypoints

keypoints1 = detect_harris_corners(gray1)
keypoints2 = detect_harris_corners(gray2)


# drawKeypoints function is used to draw keypoints
kp1 = cv2.drawKeypoints(image1, keypoints1, 0, (0, 0, 255),
                                 flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# drawKeypoints function is used to draw keypoints
kp2 = cv2.drawKeypoints(image2, keypoints2, 0, (0, 0, 255),
                                 flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


# Compute SIFT descriptors for keypoints
sift = cv2.SIFT_create()
descriptors1 = sift.compute(image1, keypoints1)[1]
descriptors2 = sift.compute(image2, keypoints2)[1]

# Create a Brute-Force Matcher
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (lower is better)
matches = sorted(matches, key=lambda x: x.distance)



# Sort the top N matches
N = 10
top_matches = matches[:N]


# Draw matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, top_matches, None)




# Show the matched image
cv2.imshow('Keypoints 1 (Harris Corners)', kp1)
cv2.imshow('Keypoints 2 (Harris Corners)', kp2)
cv2.imshow('Scale Invariant Feature Transform Matching ', matched_image)

cv2.waitKey(0)
cv2.destroyAllWindows()