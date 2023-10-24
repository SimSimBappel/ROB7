import numpy as np
import cv2

image1 = 'testImages/aau-city-1.jpg'
image2 = 'testImages/aau-city-2.jpg'

#read as grayscale
image1 = cv2.imread(image1)
image2 = cv2.imread(image2)


def doHarris(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
 
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)

    coordinates = np.argwhere(dst > 0.01 * dst.max())

    
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in coordinates]


    image1[dst>0.01*dst.max()]=[0,0,255]

    return image1, keypoints

image1, kp = doHarris(image1)
print(kp[1])

cv2.imshow('dst',image1)


if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()