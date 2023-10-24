import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()

    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Lchannel = imgHLS[:,:,1]
    #change 250 to lower numbers to include more values as "white"
    # mask = cv2.inRange(Lchannel, 240, 255)
    mask = cv2.inRange(imgHLS, np.array([0,0,0]), np.array([255,70,255]))
    res = cv2.bitwise_and(img,img, mask= mask)
    
    cv2.imshow("lol", res)
    cv2.imshow("frame", img)
    key = cv2.waitKey(1)
    if key == 27:
        print("user interrupt")
        break


cap.release()
cv2.destroyAllWindows()