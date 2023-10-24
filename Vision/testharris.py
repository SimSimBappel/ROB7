# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# """
# Difference between goodFeaturesToTrack and Harrisdetector:
# The main difference with the Harris algorithm is that you should
# specify the minimum distance between each point, the quality level
# and the number of corners to detect.

# """
# #You can use this Method to detect the Harriscorners instead of goodFeaturesToTrack :

# #dst1 = cv2.cornerHarris(gray1, 5, 7, 0.04)
# #ret1, dst1 = cv2.threshold(dst1, 0.1 * dst1.max(), 255, 0)
# #dst1 = np.uint8(dst1)
# #ret1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(dst1)
# #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# #corners1 = cv2.cornerSubPix(gray1, np.float32(centroids1), (5, 5), (-1, -1), 
# #criteria)
# #corners1 = np.int0(corners1)


# def correlation_coefficient(window1, window2):
#     product = np.mean((window1 - window1.mean()) * (window2 - window2.mean()))
#     stds = window1.std() * window2.std()
#     if stds == 0:
#         return 0
#     else:
#         product /= stds
#         return product


# window_size_width = 10
# window_size_height = 10
# lineThickness = 2

# img1 = cv2.imread('testImages/aau-city-1.jpg')
# img2 = cv2.imread('testImages/aau-city-2.jpg')
# width, height, ch = img1.shape[::]
# img2_copy = img2.copy()
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# corners1 = cv2.goodFeaturesToTrack(gray1, 30, 0.01, 5)
# corners1 = np.int0(corners1)

# corners2 = cv2.goodFeaturesToTrack(gray2, 30, 0.01, 5)
# corners2 = np.int0(corners2)

# corners_windows1 = []

# for i in corners1:
#     x, y = i.ravel()
#     cv2.circle(img1, (x, y), 3, 255, -1)

# corners_windows2 = []
# for i in corners2:
#     x, y = i.ravel()
#     cv2.circle(img2, (x, y), 3, 255, -1)

# plt.imshow(img1), plt.show()

# methods = ['SSD', 'NCC']
# for method in methods:
#     matches = []
#     for id1, i in enumerate(corners1):
#         x1, y1 = i.ravel()
#         if y1 - window_size_height < 0 or y1 + window_size_height > height or x1 - window_size_width < 0 or x1 + window_size_width > width:
#             continue
#         pt1 = (x1, y1)
#         print("pt1: ", pt1)
#         template = img1[y1 - window_size_height:y1 + window_size_height, x1 - window_size_width:x1 + window_size_width]
#         max_val = 0
#         Threshold = 1000000
#         id_max = 0
#         for id2, i in enumerate(corners2):
#             x2, y2 = i.ravel()

#             if y2 - window_size_height < 0 or y2 + window_size_height > height or x2 - window_size_width < 0 or x2 + window_size_width > width:
#                 continue
#             window2 = img2[y2 - window_size_height:y2 + window_size_height,
#                       x2 - window_size_width:x2 + window_size_width]
#             if method == 'SSD':
#                 temp_min_val = np.sum((template - window2) ** 2)
#             elif method == 'NCC':
#                 temp_min_val = correlation_coefficient(template, window2)
#             if temp_min_val < Threshold:
#                 Threshold = temp_min_val
#                 pt2 = (x2 + 663, y2)
#         matches.append((pt1, pt2))
#     stacked_img = np.hstack((img1, img2))
#     #show the first 15 matches
#     for match in matches[:15]:
#         cv2.line(stacked_img, match[0], match[1], (0, 255, 0), lineThickness)
#     matches = []
#     plt.imshow(stacked_img), plt.show()












import cv2
import numpy as np

# from skimage.io import imread
# from skimage.color import rgb2gray
import matplotlib.pyplot as plt

img = cv2.imread('testImages/aau-city-1.jpg')
# img_gray = cv2.COLOR_RGB2GRAY(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = rgb2gray(img)


img_gray = np.float32(img_gray)
#cv2.imshow("Image",img)
#cv2.imshow("Gray Image",img_gray)
#Ix = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=5)
#Iy = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=5)
kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
Ix = cv2.filter2D(img_gray,-1,kernel_x)
Iy = cv2.filter2D(img_gray,-1,kernel_y)
Ixx = Ix**2
Ixy = Ix*Iy
Iyy = Iy**2
#cv2.imshow("Ixx",Ixx)
#cv2.imshow("Iyy Image",Iyy)
#cv2.imshow("Ixy Image",Ixy)

# Loop through image and find our corners
k = 0.05

height = img_gray.shape[0]
width = img_gray.shape[1]
harris_response = []
window_size = 6
offset = int(window_size/2)
for y in range(offset, height-offset):
    for x in range(offset, width-offset):
        Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
        Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
        Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])

        # Find determinant and trace, use to get corner response
        det = (Sxx * Syy) - (Sxy ** 2)
        trace = Sxx + Syy
        r = det - k * (trace ** 2)

        harris_response.append([x, y, r])
img_copy = np.copy(img)
thresh = 500
#sift = cv2.xfeatures2d.SIFT_create()
#kp,dc = sift.compute(img,None)
for response in harris_response:
    x, y, r = response
    if r > thresh:
        img_copy[y, x] = [255, 0, 0]

plt.imshow(img_copy)
cv2.waitKey(0)
plt.show()