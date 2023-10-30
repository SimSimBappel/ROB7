import cv2
import numpy as np



img = cv2.imread("/home/simon/fs_cones_val/val_images/ulm_00009.jpg")

x,y = img.shape[1::-1]
with open("/home/simon/fs_cones_val/val/ulm_00009.txt", "r") as file:
    for line in file:
        words = line.split()
        # print(line)
        coords = [float(word) for word in words[1:]]
        if words[0] == "0":
            print("Blue")            
            x1 = int(coords[0] * x)
            y1 = int(coords[1] * y)
            x2 = int(coords[2] * x/2)
            y2 = int(coords[3] * y/2)
            cv2.rectangle(img, (x1-x2, y1-y2), (x1+x2, y1+y2), (255,0,0), 1)
        elif words[0] == "1":
            print("yellow")
            x1 = int(coords[0] * x)
            y1 = int(coords[1] * y)
            x2 = int(coords[2] * x/2)
            y2 = int(coords[3] * y/2)
            cv2.rectangle(img, (x1-x2, y1-y2), (x1+x2, y1+y2), (0,255,255), 1)
        else:
            continue

file.close()

while True:


    
    
    
    
    
    
    cv2.imshow("Output", img)
    key = cv2.waitKey(1)
    if key == 27:
        print("user interrupt")
        break


cv2.destroyAllWindows()