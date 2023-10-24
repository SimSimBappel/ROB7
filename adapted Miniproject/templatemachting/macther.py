import cv2
import numpy as np
# import imutils

# image = cv2.imread("/home/simon/Pictures/Webcam/newcone.jpg")
template = cv2.imread("/home/simon/Pictures/Webcam/cone.png")
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture(0)

while True: 
    _,image = cap.read()
    cv2.imshow("Image", image)
    cv2.imshow("Template", template)
    # convert both the image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    w, h = templateGray.shape[::-1]

    res = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)

    thresh = 0.7

    loc = np.where(res >= thresh)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

    cv2.imshow("frame", res)



    # # Resize the image according to scale and 
    # # keeping track of ratio of resizing 
    # resize = imutils.resize(imageGray, width=int(shape[0]), height=int(imageGray.shape[1]*scale))
    
    # # If resize image is smaller than that of template 
    # # break the loop 
    # # Detect edges in the resized, grayscale image and apply template 
    # # Matching to find the template in image edged 
    # # If we have found a new maximum correlation value, update 
    # # the found variable if 
    # # found = null/maxVal > found][0] 
    # if resized.shape[0] < h or resized.shape[1] < w: 
    #         break
    # found=(maxVal, maxLoc, r) 
    
    # # Unpack the found variables and compute(x,y) coordinates 
    # # of the bounding box 
    # (__, maxLoc, r)=found 
    # (startX, startY)=(int(maxLoc[0]*r), int maxLoc[1]*r) 
    # (endX, endY)=(int((maxLoc[0]+tw)*r), int(maxLoc[1]+tH)*r) 
    
    # # Draw a bounding box around the detected result and display the image 
    # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2) 
    # cv2.imshow("Image", image) 






    # # perform template matching
    # result = cv2.matchTemplate(imageGray, templateGray,
    #     cv2.TM_CCOEFF_NORMED)
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

    # # determine the starting and ending (x, y)-coordinates of the
    # # bounding box
    # (startX, startY) = maxLoc
    # endX = startX + template.shape[1]
    # endY = startY + template.shape[0]


    # # draw the bounding box on the image
    # cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
    # # show the output image





    cv2.imshow("Output", image)
    key = cv2.waitKey(1)
    if key == 27:
        print("user interrupt")
        break


cap.release()
cv2.destroyAllWindows()