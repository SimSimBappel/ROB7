from concurrent.futures import process
from pyexpat import features
from matplotlib import pyplot as plt
import statistics
import histogram as hist
import numpy as np
import cv2 
import os
import time
import scipy.optimize


class Pipeline(object):
    """
    The main image processing pipeline for detecting traffic signs:
    Image acquisition -> Preprocessing -> Segmentation -> Representation -> Classification
    """

    def __init__(self):
        """
        Constructor - initializes the instantiated objects variables
        """
        # Dataset 0=blue 1=yellow
        self.imagenames = []
        # Image variables
        self.images = []
        self.image_path = "/home/simon/fs_cones"
        self.font = cv2.FONT_HERSHEY_COMPLEX
        # self.templateGray = cv2.imread("/home/simon/cone_template.png", cv2.COLOR_BGR2GRAY)
        self.template = cv2.imread("/home/simon/cone_template3.png")
        self.templateGray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)


        # Boolean operator for debugging -> Can be changed to display processing steps
        # (Warning, this shows all masks and images + print()'s per iteration)
        self.debugging = False

        # Thresholding variables (HSV Colorspace) -> using the HsvRangeTool in "Tools", to obtain values.
        # self.red1_high  = (8, 255, 255)
        # self.red1_low   = (0, 130, 0)
        # self.red2_high  = (179, 255, 255)
        # self.red2_low   = (172, 50, 0)
        # self.blue_high = (122, 255, 255)
        # self.blue_low = (101, 76, 0)
        # self.white_high = (179, 100, 255)
        # self.white_low = (0, 0, 70)

        # Thresholds for cones HSV colorspace
        self.yellow_low= (23, 51, 176) 
        self.yellow_high = (81, 243, 255)

        self.blue_low = (77, 84, 64)
        self.blue_high = (125, 255, 255)

        # Threshold for HSL (HLS)
        self.white_low = (0,240,0)
        self.white_high = (255,255,255)
        self.black_low = (0,0,0)
        self.black_high = (255,0,255) # 50


        self.yellow_mask = 0
        self.black_mask = 0
        self.blue_mask = 0
        self.white_mask = 0
        # self.red_concatenated_mask = 0
        self.yellow_cone_mask = 0
        self.blue_cone_mask = 0
        # self.combined_mask = 0
        self.mask_processed = False

        # self.red1_filtered = 0
        # self.red2_filtered = 0
        self.blue_filtered = 0
        self.white_filtered = 0

        self.area_storage = []
        
        # self.red_contours = 0
        self.yellow_contours = 0
        self.blue_contours = 0
        

        # Filtering OG
        # self.square_kernel = np.ones((8, 8), np.uint8)
        # self.circular_kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        # self.circular_kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        # self.circular_kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

        # Filter updated
        self.square_kernel = np.ones((8, 8), np.uint8)
        self.circular_kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        self.circular_kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        self.circular_kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

        self.opening = np.ndarray
        self.closing = np.ndarray

        # Blobs and features
        # self.filtered_blobs = []
        self.filtered_blobs_yellow = []
        self.filtered_blobs_blue = []
        # self.features = []
        self.yellow_features = []
        self.blue_features = []

        self.bboxes = []

    def reset(self):
        # self.red1_mask = 0
        # self.red2_mask = 0
        self.blue_mask = 0
        self.white_mask = 0
        # self.red_concatenated_mask = 0
        self.yellow_cone_mask = 0
        self.blue_cone_mask = 0
        # self.combined_mask = 0
        self.mask_processed = False

        # self.red1_filtered = 0
        # self.red2_filtered = 0
        self.blue_filtered = 0
        self.white_filtered = 0

        self.area_storage = []
        
        # self.red_contours = 0
        self.yellow_contours = 0
        self.blue_contours = 0
        

        self.opening = np.ndarray
        self.closing = np.ndarray

        # Blobs and features
        # self.filtered_blobs = []
        self.filtered_blobs_yellow = []
        self.filtered_blobs_blue = []
        # self.features = []
        self.yellow_features = []
        self.blue_features = []

        self.bboxes = []


    def load_images_from_folder(self):
        for filename in os.listdir(self.image_path):
            temp_img = cv2.imread(os.path.join(self.image_path,filename))
            # temp_img = cv2.resize(temp_img, None, fx=0.30, fy=0.30)
            if temp_img is not None:
                self.imagenames.append(filename)
                self.images.append(temp_img)
        return
    
    def get_images(self, index):
        return self.images[index]
    
    def get_imagename(self, i):
        return self.imagenames[i]

    def get_all_images(self):
        return self.images

    # def get_area_storage(self):
    #     return self.area_storage

    def get_mask(self, index):
        if (self.mask_processed == True):
            if (index == 0):
                # return self.red1_mask
                return self.yellow_mask
            if (index == 1):
                # return self.red2_mask
                return self.blue_mask
            if (index == 2):
                # return self.blue_mask
                return self.white_mask  
            if (index == 3):
                # return self.white_mask  
                return self.black_mask
            if(index>3):
                print("[",self.get_mask.__name__,"]", "> ERROR: Out of range, please use parameter 0 (red mask) or 1 (blue mask)")
        else: 
            print("[",self.get_mask.__name__,"]", " > ERROR: Please run the thresholding function before calling get_mask()")
        
    def CLAHE_equalization(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)

        if (self.debugging == True):
            cv2.imshow("CLAHE Equalized", equalized)

        return #equalized

    # def plot_histogram(self, image):
    #     hist,bins = np.histogram(image.flatten(),256,[0,256])
    #     cdf = hist.cumsum()
    #     cdf_normalized = cdf * float(hist.max()) / cdf.max()
    #     plt.plot(cdf_normalized, color = 'b')
    #     plt.hist(image.flatten(),256,[0,256], color = 'r')
    #     plt.xlim([0,256])
    #     plt.legend(('cdf','histogram'), loc = 'upper left')
    #     plt.show()

    def thresholding(self, image):
        '''
        Thresholding based on the inRange() function
        Convert images to HSV space -> Set the HSV threshold to the blue and red range. -> Mask the blue and red objects
        
        Source: https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html 
        '''
        # Change to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if (self.debugging == True):
            cv2.imshow("HSV Image", hsv)

        # Build masks based on thresholds
        # We do two red mask for combining the upper and lower threshold of the HSV colorspace (These are combined later)
        self.yellow_mask = cv2.inRange(hsv, self.yellow_low, self.yellow_high)
        self.blue_mask = cv2.inRange(hsv, self.blue_low, self.blue_high)



        # Change to HSL color space
        # hsl = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
        # self.black_mask = cv2.inRange(hsl, self.black_low, self.black_high)
        # self.white_mask = cv2.inRange(hsl, self.white_low, self.white_high)
        self.mask_processed = True

        # Test output
        if (self.debugging == True):
            cv2.imshow("Yellow mask", self.yellow_mask)
            # cv2.imshow("Black mask", self.black_mask)
            cv2.imshow("Blue mask", self.blue_mask)
            # cv2.imshow("White mask", self.white_mask)

        return 

    def pre_processing(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        R, G, B = cv2.split(image)

        equalized_R = clahe.apply(R)
        equalized_G = clahe.apply(G)
        equalized_B = clahe.apply(B)

        # Test these methods
        CLAHE_image = cv2.merge((equalized_R, equalized_G, equalized_B))
        edited = cv2.convertScaleAbs(image, alpha=0.9, beta=30)

        CLAHE_hist  = hist.hist_curve(CLAHE_image)
        edited_hist = hist.hist_curve(edited)

        if (self.debugging == True):
            cv2.imshow("CLAHE Processed", CLAHE_image)
            cv2.imshow("Brightness / contrast correction", edited)
            cv2.imshow('CLAHE [Histogram]', CLAHE_hist)
            cv2.imshow('Further edited [histogram]', edited_hist)

        return edited

    def morphology_filter(self, mask):
        if mask is None:
            print("WARNING: The mask is empty - this can also be caused by zero objects")
            return

        else:
            # In median blurring, the central element is always replaced by some pixel value in the image. 
            # It reduces the noise effectively.
            median = cv2.medianBlur(mask, 3)

            # Opening is erosion followed by dilation. (Removes small connected components and small protrusions)
            # Closing is dilation followed by erosion. (fills in small holes and gaps between connected components)
            closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, self.circular_kernel_tiny)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, self.circular_kernel_small)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.circular_kernel_large)

            return opening

    def white_object_mask(self, image, cnt):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros_like(hsv) # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, [cnt], 0, (255,255,255), -2)
        out = np.zeros_like(hsv) # Extract out the object and place into output image
        out[mask == (255,255,255)] = hsv[mask == (255,255,255)]
        self.white_mask = cv2.inRange(out, self.white_low, self.white_high)
        return

    def detect_shape(self, cnt):
        shape = ""
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Triangle
        if len(approx) == 3:
            shape = "triangle"

        # Square or rectangle
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # A square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # Pentagon
        elif len(approx) == 5:
            shape = "pentagon"

        # Otherwise assume as circle or oval
        else:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "circle" if ar >= 0.95 and ar <= 1.05 else "oval"

        return cnt, shape, peri, approx

    def classify(self, image):
        imgw, imgh = image.shape[1::-1]
    
        # Iterates through every object
        for i in range(0, len(self.filtered_blobs_yellow)):
            features = self.yellow_features[i]     # Gets the feature vector in question
            blob = self.filtered_blobs_yellow[i]   # Gets the corresponding object
            # Gets and draws the bounding box of the object on the original image
            x, y, w, h = cv2.boundingRect(blob)
            h_int = int(h*1.3)
            # cropped_image = self.crop_blob(x, y, h, w, h_int)
        
            # Classification process
            # 0: shape, 1: circularity, 2: aspect ratio, 3: compactness, 4: color_mean, 5: yellow_percent, 6: black_percent, 7: distance to COM (bbox)

            # if features[2] < 0.85 and features[2] > 0.65 and features[5] > 39: #(features[0] == 'triangle' or features[0] == 'pentagon') and
            #     # print(features[4])
            #     text = "Yellow cone: "
            #     cv2.putText(image, text , (x-10, y-10), self.font, 0.5, (0,255,255), 1, cv2.LINE_AA)
            if y < h_int:
                cropped_image = self.yellow_filtered[y-h:y+h, x:x+w]

            else:
                cropped_image = self.yellow_filtered[y-h_int:y+h, x:x+w]



            if features[2] < 2.1 and features[2] > 1.2 and features[5] > 30:
                res = self.template_match(cropped_image)
                if res > 0.6:    
                    #text = "Blue bottom: " + str(round(features[5],2))
                    # text = "res: " + str(res)
                    # cv2.putText(image, text, (x-10, y-h-10), self.font, 0.5, (255,0,255), 1, cv2.LINE_AA)
                    # cv2.rectangle(image, (x, y-h_int), (x+w, y+h), (0,255, 255), 1)
                    #format "(color) (centerx) (centery) (w) (h)"
                    # bbox = "1 " + str((x-w/2)/imgw) + " " + str((y-h/2)/imgh) + " " + str(w/imgw) + " " + str(h/imgh)
                    bbox = "1 " + str(x) + " " + str(y-h_int) + " " + str(w+x) + " " + str(h+y)
                    self.bboxes.append(bbox)

                else:
                    continue
                    # text = "res: " + str(res)
                    # cv2.putText(image, text, (x-15, y-10), self.font, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y-h), (x+w, y+h), (0,0,255), 1)
                    #print(res)
                
            else:
                # text = str(features[0]) + "AR: " + str(round(features[2],2)) + "yellow: " + str(round(features[5],2)) + "blue:" + str(round(features[6],2))
                # cv2.putText(image, text, (x-15, y-10), self.font, 0.5, (0,0,0), 1, cv2.LINE_AA)
                text ="AR: " + str(round(features[2],2)) + "yellow:" + str(round(features[5],2))
                # cv2.putText(image, text, (x-30, y), self.font, 0.5, (0,0,0), 1, cv2.LINE_AA)
                # cv2.rectangle(image, (x, y-h), (x+w, y+h), (255,255,0), 1)
                

                if (self.debugging == True):
                    print("Unknown object")



        for i in range(0, len(self.filtered_blobs_blue)):
            
            features = self.blue_features[i]     # Gets the feature vector in question
            blob = self.filtered_blobs_blue[i]   # Gets the corresponding object
            # Gets and draws the bounding box of the object on the original image
            x, y, w, h = cv2.boundingRect(blob)
            h_int = int(h*1.3)
           
            # cropped_image = self.crop_blob(x, y, h, w, h_int)
            if y < h_int:
                cropped_image = self.blue_filtered[y-h:y+h, x:x+w]

            else:
                cropped_image = self.blue_filtered[y-h_int:y+h, x:x+w]

            # Classification process
            # 0: shape, 1: circularity, 2: aspect ratio, 3: compactness, 4: color_mean, 5: blue_percent, 6: white_percent, 7: distance to COM (bbox)
            # Finding cone bottoms 
            if features[2] < 2.1 and features[2] > 1.2 and features[5] > 30: # (features[0] == 'triangle' or features[0] == 'pentagon') and
                res= self.template_match(cropped_image)

                if res > 0.6:    
                    #text = "Blue bottom: " + str(round(features[5],2))
                    # text = "res: " + str(res)
                    # cv2.putText(image, text, (x-10, y-h-10), self.font, 0.5, (255,0,255), 1, cv2.LINE_AA)
                    # cv2.rectangle(image, (x, y-h_int), (x+w, y+h), (255,0,0), 1)

                    #format "(color) (centerx) (centery) (w) (h)"
                    # bbox = "0 " + str((x-w/2)/imgw) + " " + str((y-h/2)/imgh) + " " + str(w/imgw) + " " + str(h/imgh)
                    bbox = "0 " + str(x) + " " + str(y-h_int) + " " + str(w+x) + " " + str(h+y)
                    self.bboxes.append(bbox)


                else:
                    continue
                    # text = "res: " + str(res)
                    # cv2.putText(image, text, (x-15, y-10), self.font, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y-h), (x+w, y+h), (0,0,255), 1)
                    # print(res)

                

            else:
                text ="AR: " + str(round(features[2],2)) + "blue:" + str(round(features[5],2))
                # print(text)
                # cv2.putText(image, text, (x-30, y), self.font, 0.5, (0,0,0), 1, cv2.LINE_AA)
                # cv2.rectangle(image, (x, y-h), (x+w, y+h), (255,255,0), 1)
                
                if (self.debugging == True):
                    print("Unknown object")
        return image, self.bboxes, imgw, imgh
    

    # def crop_blob(self, x, y, h, w, h_int):
    #     if y < h_int:
    #         cropped_image = self.blue_filtered[y-h:y+h, x:x+w]

    #     else:
    #         cropped_image = self.blue_filtered[y-h_int:y+h, x:x+w]

    #     return cropped_image


    def template_match(self, crop):  
        w, h = crop.shape[::-1]
        template = cv2.resize(self.templateGray, (w,h))



        crop = np.float32(crop)
        template = np.float32(template)

        res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        
        # thresh = 0.7

        # loc = np.where(res >= thresh)

        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(crop, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

        # cv2.imshow("frame", template)
        return res#, template

    def get_segment_crop(self, img,tol=0, mask=None):
        if mask is None:
            mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    def feature_extraction(self, cnt, shape, image, color):
        if color == 'yellow':
            feature_vector = []
            # Compute image moments
            moments = cv2.moments(cnt)
            area = moments['m00']

            # Get length of blob edge
            perimeter = cv2.arcLength(cnt, True)

            # Circularity of blob
            circularity = 4*np.pi*area/(perimeter**2)

            # Get bounding box
            box_x, box_y, w, h = cv2.boundingRect(cnt)

            # Width/height ratio
            aspect_ratio = w/h

            compactness = area/(w*h)

            # Percent of isolated colors in ROI
            ROI_area = w*h

            yellow_crop = self.yellow_mask[int(box_y):int(box_y+h), int(box_x):int(box_x+w)]
            if yellow_crop is not None: 
                yellow_pixels = cv2.countNonZero(yellow_crop)
                yellow_percent = (yellow_pixels / ROI_area) * 100

                if (self.debugging == True):
                    cv2.imshow('Yellow percent mask', yellow_crop)
                    print("Yellow (%): ", yellow_percent)

            # black_crop = self.black_mask[int(box_y):int(box_y+h), int(box_x):int(box_x+w)]
            # if black_crop is not None:
            #     black_pixels = cv2.countNonZero(black_crop)
            #     black_percent = (black_pixels / ROI_area) * 100

            #     if (self.debugging == True):
            #         cv2.imshow('Black percent mask', black_crop)
            #         print("Black (%): ", black_percent)
            
            # Average of each color channel inside the contour
            mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            color_mean = cv2.mean(image, mask=mask)

            # Distance from COM to center of bounding box
            com = [int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])]
            com_distance = np.sqrt((com[0]-box_x)**2 + (com[1]-box_y)**2)/(w*h)

            feature_vector = [shape,
                                circularity,
                                aspect_ratio,
                                compactness,
                                color_mean,
                                yellow_percent,
                                # black_percent,
                                com_distance]

            self.yellow_features.append(feature_vector)

            if (self.debugging == True):
                print("Feature vector: ", feature_vector)

            return
        

        else: 
            feature_vector = []
            # Compute image moments
            moments = cv2.moments(cnt)
            area = moments['m00']

            # Get length of blob edge
            perimeter = cv2.arcLength(cnt, True)

            # Circularity of blob
            circularity = 4*np.pi*area/(perimeter**2)

            # Get bounding box
            box_x, box_y, w, h = cv2.boundingRect(cnt)

            # Width/height ratio
            aspect_ratio = w/h

            compactness = area/(w*h)

            # Percent of isolated colors in ROI
            ROI_area = w*h

            blue_crop = self.blue_mask[int(box_y):int(box_y+h), int(box_x):int(box_x+w)]
            if blue_crop is not None:
                blue_pixels = cv2.countNonZero(blue_crop)
                blue_percent = (blue_pixels / ROI_area) * 100

                if (self.debugging == True):
                    cv2.imshow('Blue percent mask', blue_crop)
                    print("Blue (%): ", blue_percent)

            # white_crop = self.white_mask[int(box_y):int(box_y+h), int(box_x):int(box_x+w)]
            # if white_crop is not None:
            #     white_pixels = cv2.countNonZero(white_crop)
            #     white_percent = (white_pixels / ROI_area) * 100

            #     if (self.debugging == True):
            #         cv2.imshow('White percent mask', white_crop)
            #         print("White (%): ", white_percent)
            
            # Average of each color channel inside the contour
            mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            color_mean = cv2.mean(image, mask=mask)

            # Distance from COM to center of bounding box
            com = [int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])]
            com_distance = np.sqrt((com[0]-box_x)**2 + (com[1]-box_y)**2)/(w*h)

            feature_vector = [shape,
                                circularity,
                                aspect_ratio,
                                compactness,
                                color_mean,
                                blue_percent,
                                # white_percent,
                                com_distance]

            self.blue_features.append(feature_vector)

            if (self.debugging == True):
                print("Feature vector: ", feature_vector)

            return

    def blob_extraction(self, image, mask, color):
        if mask is not None:

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0:
                for cnt in contours:       

                    area = cv2.contourArea(cnt)
                    self.area_storage.append(area)   
                    #print(area)

                    # We filter some of the noise that persists in the filtered binary masks
                    if area > 30:  #920  

                        processed_cnt, shape, peri, approx = self.detect_shape(cnt)
                        
                        if color == 'yellow':
                            self.filtered_blobs_yellow.append(processed_cnt)
                        else: 
                            self.filtered_blobs_blue.append(processed_cnt)
                        
                        self.feature_extraction(processed_cnt, shape, image, color)

                        # Find the biggest countour (c) by the area
                        #c = max(contours, key = cv2.contourArea)

                        x,y,w,h = cv2.boundingRect(processed_cnt)

                        # Draw bounding box in green
                        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

            if (self.debugging == True):
                cv2.imshow('Objects', image)

    def get_objects(self, image):
        self.yellow_filtered = self.morphology_filter(self.get_mask(0))
        self.blue_filtered = self.morphology_filter(self.get_mask(1))
        # self.white_filtered = self.morphology_filter(self.get_mask(2))
        # self.black_filtered = self.morphology_filter(self.get_mask(3))

        # self.red_concatenated_mask = self.red1_filtered+self.red2_filtered

        #self.combined_mask = self.yellow_filtered + self.blue_filtered + self.white_filtered + self.black_filtered

        self.yellow_cone_mask = self.yellow_filtered# + self.white_filtered
        self.blue_cone_mask = self.blue_filtered# + self.black_filtered 

        self.yellow_contours = self.blob_extraction(image, self.yellow_cone_mask, 'yellow')
        self.blue_contours = self.blob_extraction(image, self.blue_cone_mask, 'blue')

        #cv2.imshow("all blobs", self.combined_mask)

       # if (self.debugging == True):
        # if self.yellow_filtered is not None:
        #     cv2.imshow('Filtered yellow', self.yellow_filtered)
        # if self.blue_filtered is not None:
        #     cv2.imshow('Filtered blue', self.blue_filtered)
        # if self.white_filtered is not None:
        #     cv2.imshow('Filtered white', self.white_filtered)
        # if self.black_filtered is not None:
        #     cv2.imshow('Filtered black', self.black_filtered)

        return


# def search_file(filename, folder_path):
#     for root, dirs, files in os.walk(folder_path):
#         if filename in files:
#             return os.path.join(root, filename)


def bbox_iou(boxA, boxB):
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.
    
  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])


#   print(xA)
#   print(xB)

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou


def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''

    n_true = len(bbox_gt)
    n_pred = len(bbox_pred)
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # print(bbox_gt[1])
    # print(" ")
    # print(bbox_pred)

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            # iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])
            iou_matrix[i, j] = bbox_iou(bbox_gt[i], bbox_pred[j])
            continue

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 


def process_images(instance): #, cap, img
    #image folder
    images = instance.get_all_images()
    IoU = 0
    totalIoUentries = 0
    for i, img in enumerate(images):
        # timer = time.perf_counter()
        print("NEXT")
        instance.reset()
        image = instance.pre_processing(img)
        instance.thresholding(image)
        instance.get_objects(image)
        classified, bboxes, imgw, imgh = instance.classify(image)
        # timer = time.perf_counter() - timer        
        # text = "Time to run: " + str(round(timer,3)) + "s, aka: " + str(round(1/timer,3)) + "Hz"
        # print(text)

        # IoU calculation
        bgtbboxes = []
        ygtbboxes = []
        bbboxes = []
        ybboxes = []
        blueCones = 0
        yellowCones = 0
        blueConesDetected = 0
        yellowConesDetected = 0
        AvgIoUImg = 0
        accept = False
        filename = "/home/simon/fs_cones_val/val/" + instance.get_imagename(i)
        filename = filename[:-3] + "txt"
        with open(filename, "r") as file:
            for line in file:
                words = line.split()
                gtcoords = [float(word) for word in words[1:]]
                # if words[0] == "0":
                #     bgtbboxes.append(gtcoords)
                #     accept = True
                # elif words[0] == "1":
                #     ygtbboxes.append(gtcoords)
                #     accept = True         
                if words[0] == "0":
                    # print("Blue")            
                    x1 = int(gtcoords[0] * imgw)
                    y1 = int(gtcoords[1] * imgh)
                    x2 = int(gtcoords[2] * imgw/2)
                    y2 = int(gtcoords[3] * imgh/2)
                    # tempbox = [x1, y1, x2, y2]
                    tempbox = [x1-x2, y1-y2, x1+x2, y1+y2]
                    bgtbboxes.append(tempbox)
                    blueCones += 1
                    accept = True
                    cv2.rectangle(classified, (x1-x2, y1-y2), (x1+x2, y1+y2), (0,0,0), 1)
                elif words[0] == "1":
                    # print("yellow")
                    x1 = int(gtcoords[0] * imgw)
                    y1 = int(gtcoords[1] * imgh)
                    x2 = int(gtcoords[2] * imgw/2)
                    y2 = int(gtcoords[3] * imgh/2)
                    # tempbox = [x1, y1, x2, y2]
                    tempbox = [x1-x2, y1-y2, x1+x2, y1+y2]
                    ygtbboxes.append(tempbox)
                    yellowCones += 1
                    accept = True
                    cv2.rectangle(classified, (x1-x2, y1-y2), (x1+x2, y1+y2), (0,0,0), 1)     
        file.close()   

        for entry in bboxes:
            nums = entry.split()
            coords = [int(num) for num in nums[1:]]
            if nums[0] == "0":
                bbboxes.append(coords)
                # blueConesDetected += 1
                cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (255,0,0), 1)
            elif nums[0] == "1":
                # yellowConesDetected +=1
                ybboxes.append(coords)
                cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (0,255,255), 1)

        if accept:
            # print(bbboxes)
            # print(ybboxes)
            # print(bgtbboxes)
            # print(ygtbboxes)

            blueconfusius = match_bboxes(bgtbboxes, bbboxes)
            print(blueconfusius)
            yellowconfusius = match_bboxes(ygtbboxes, ybboxes)
            # print(yellowconfusius[2])


            for x in blueconfusius[2]:
                AvgIoUImg += x
                totalIoUentries += 1

            for x in yellowconfusius[2]:
                AvgIoUImg += x
                totalIoUentries += 1

            
            
            if 0 < (len(blueconfusius[2]) + len(yellowconfusius[2])):
                IoU += AvgIoUImg
                AvgIoUImg = AvgIoUImg / (len(blueconfusius[2]) + len(yellowconfusius[2]))

                print("Avg IoU on image " + str(round(AvgIoUImg, 3)) + "% Cones detected/GT: " + str(len(blueconfusius[0])+len(yellowconfusius[0])) + "/" + str(yellowCones+blueCones))



            

        cv2.imshow(instance.get_imagename(i), classified)  

        k = cv2.waitKey(0)
        if k == 27: #esc
            cv2.destroyAllWindows() 
            break
        
        cv2.destroyAllWindows() 

    IoU = IoU / totalIoUentries
    print("IoU: " + str(IoU))







    # #webcam
    # instance.reset()
    # image = instance.pre_processing(img)
    # instance.thresholding(img)
    # instance.get_objects(img)
    # classified = instance.classify(img)

    # cv2.imshow('Classifier', classified)

   
    # k = cv2.waitKey(5) & 0xFF
    # if k == 27: #esc
    #     cap.release()
    #     cv2.waitKey(0) 
    #     cv2.destroyAllWindows() 


# Main function. Run this to classify. Change path in get_image_paths() to your path
def main():
    p = Pipeline()

    #image folder

    p.load_images_from_folder()
    process_images(p)

    

    # #webcam
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     _, image = cap.read()
    #     process_images(p,cap,image)
    # cap.release()
    # cv2.destroyAllWindows() 


if __name__ == '__main__':
    main()