import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# Function for drawing lines in an image
def draw_lines(img, lines, colors):
    r,c = img.shape
    scale = r/100.0 # scale line width based on image size
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for k,r in enumerate(lines):
        color = colors[k]
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color, int(scale))
    return img

# Function for drawing points in an image
def draw_points(img, pts, colors):
    r,c = img.shape
    scale = r/100.0 # scale point size based on image size
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for k,pt in enumerate(pts):
        color = colors[k]
        print(pt)
        img = cv2.circle(img,(int(pt[0]),int(pt[1])),int(scale*3),color, int(scale))
    return img

# Init our fundamental matrix from exercise 3
F = np.array([[-3.84318408e-09,  2.85542943e-06, -1.87766466e-03],
              [-2.22183625e-06, -5.94016361e-08, -9.60318584e-02],
              [ 1.36631737e-03,  9.68941860e-02,  1.00000000e+00]])

# Load the images
left_img = cv2.imread('Lec8/opencv-samples/left/left02.jpg')
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)

right_img = cv2.imread('Lec8/opencv-samples/right/right02.jpg')
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Carefully (not really) manually selected points from the left image
left_pts = np.array([[256.0, 362.0],
	             [292.0, 343.0],
	             [334.0, 327.0],
	             [381.0, 308.0],
	             [434.0, 286.0],
	             [493.0, 262.0]])

# Find epilines in the right image correponding to these points from the left image
img_index = 1 # index of the image (1 or 2) containing the points, in this case imgL = 1
linesR = cv2.computeCorrespondEpilines(left_pts.reshape(-1,1,2), img_index, F)
linesR = linesR.reshape(-1,3)

colors = [tuple(np.random.randint(0,255,3).tolist()) for i in left_pts]
imgL_points = draw_points(left_gray, left_pts, colors)
imgR_lines = draw_lines(right_gray, linesR, colors)

ax = plt.subplot(121)
ax.set_title('left image')
ax.imshow(imgL_points)

ax = plt.subplot(122)
ax.set_title('right image')
ax.imshow(imgR_lines)
plt.show()
# The epilines in the right image appears to correspond
# well with the lines in the left image. Not 100% as
# there is some noise. Perhaps the calibration could
# improve with some more images
