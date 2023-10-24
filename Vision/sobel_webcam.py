import cv2 as cv2

cap = cv2.VideoCapture(0)


scale = 1
delta = 0
ddepth = cv2.CV_16S
ddepth = cv2.CV_64F

while(1):
    _, img = cap.read()


    src = cv2.GaussianBlur(img, (5, 5), 0)


    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

   
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)



     #same thing just rgb
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)

    grad_x = cv2.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)


    grad2 = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    

    cv2.imshow('raw', img)
    cv2.imshow('sobel_grey', grad)
    cv2.imshow('sobel_RGB', grad2)
    k = cv2.waitKey(5) & 0xFF
    if k == 27: #esc
        break
	

cv2.destroyAllWindows()

#release the frame
cap.release()
