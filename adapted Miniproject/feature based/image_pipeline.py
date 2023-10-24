from concurrent.futures import process
from pyexpat import features
from matplotlib import pyplot as plt
import statistics
import histogram as hist
import numpy as np
import cv2 
import os


class Pipeline(object):
    """
    The main image processing pipeline for detecting traffic signs:
    Image acquisition -> Preprocessing -> Segmentation -> Representation -> Classification
    """

    def __init__(self):
        """
        Constructor - initializes the instantiated objects variables
        """
        # Source: https://en.wikipedia.org/wiki/Road_signs_in_Denmark
        self.sign_types = [
            "local_speed_limit", "end_of_local_speed_limit", "one_way", "no_waiting", "no_waiting_arrow", "no_waiting_zone", "yield", "parking", "parking_arrow", "no_left_turn", "mandatory_direction_right", "arrow_direction_sign", "roadwork_area", "instructions_for_disabled", "no_truck"
                          ]
        
        # Image variables
        self.images = []
        self.image_path = os.getcwd() + "/miniproject/sign_detection-main/tsign_dataset/images/evening"
        self.font = cv2.FONT_HERSHEY_COMPLEX

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
        self.blue_low = (90, 59, 217)
        self.blue_high = (131, 255, 255)


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
        

        # Filtering
        self.square_kernel = np.ones((8, 8), np.uint8)
        self.circular_kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        self.circular_kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        self.circular_kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

        self.opening = np.ndarray
        self.closing = np.ndarray

        # Blobs and features
        # self.filtered_blobs = []
        self.filtered_blobs_yellow = []
        self.filtered_blobs_blue = []
        # self.features = []
        self.yellow_features = []
        self.blue_features = []

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


    def load_images_from_folder(self):
        for filename in os.listdir(self.image_path):
            temp_img = cv2.imread(os.path.join(self.image_path,filename))
            temp_img = cv2.resize(temp_img, None, fx=0.30, fy=0.30)
            if temp_img is not None:
                self.images.append(temp_img)
        return
    
    def get_images(self, index):
        return self.images[index]

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
        hsl = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
        self.black_mask = cv2.inRange(hsl, self.black_low, self.black_high)
        self.white_mask = cv2.inRange(hsl, self.white_low, self.white_high)
        self.mask_processed = True

        # Test output
        if (self.debugging == True):
            cv2.imshow("Yellow mask", self.yellow_mask)
            cv2.imshow("Black mask", self.black_mask)
            cv2.imshow("Blue mask", self.blue_mask)
            cv2.imshow("White mask", self.white_mask)

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
            median = cv2.medianBlur(mask, 1)

            # Opening is erosion followed by dilation. (Removes small connected components and small protrusions)
            # Closing is dilation followed by erosion. (fills in small holes and gaps between connected components)
            closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, self.circular_kernel_tiny)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, self.circular_kernel_small)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.circular_kernel_large)

            return closing

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
        # Iterates through every object
        for i in range(0, len(self.filtered_blobs_yellow)):
            features = self.yellow_features[i]     # Gets the feature vector in question
            blob = self.filtered_blobs_yellow[i]   # Gets the corresponding object
            # Gets and draws the bounding box of the object on the original image
            x, y, w, h = cv2.boundingRect(blob)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,0), 1)
            

            # Classification process
            # 0: shape, 1: circularity, 2: aspect ratio, 3: compactness, 4: color_mean, 5: yellow_percent, 6: black_percent, 7: distance to COM (bbox)

            if (features[0] == 'triangle' or features[0] == 'pentagon') and features[2] < 0.85 and features[2] > 0.65 and features[5] > 39:
                # print(features[4])
                text = "Yellow cone: "
                cv2.putText(image, text , (x-10, y-10), self.font, 0.5, (0,255,255), 1, cv2.LINE_AA)

            else:
                text = str(features[0]) + "AR: " + str(round(features[2],2)) + "yellow: " + str(round(features[5],2)) + "blue:" + str(round(features[6],2))
                cv2.putText(image, text, (x-15, y-10), self.font, 0.5, (0,0,0), 1, cv2.LINE_AA)
                
                if (self.debugging == True):
                    print("Unknown object")


        for i in range(0, len(self.filtered_blobs_blue)):
            features = self.blue_features[i]     # Gets the feature vector in question
            blob = self.filtered_blobs_blue[i]   # Gets the corresponding object
            # Gets and draws the bounding box of the object on the original image
            x, y, w, h = cv2.boundingRect(blob)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,0), 1)
            

            # Classification process
            # 0: shape, 1: circularity, 2: aspect ratio, 3: compactness, 4: color_mean, 5: blue_percent, 6: white_percent, 7: distance to COM (bbox)
        
            if (features[0] == 'triangle' or features[0] == 'pentagon') and features[2] < 0.85 and features[2] > 0.65 and features[5] > 30:
                # print(features[4])
                text = "Blue cone: "
                cv2.putText(image, text, (x-10, y-10), self.font, 0.5, (255,0,0), 1, cv2.LINE_AA)


            else:
                text = str(features[0]) + "AR: " + str(round(features[2],2)) + "yellow: " + str(round(features[5],2)) + "blue:" + str(round(features[6],2))
                cv2.putText(image, text, (x-15, y-10), self.font, 0.5, (0,0,0), 1, cv2.LINE_AA)
                
                if (self.debugging == True):
                    print("Unknown object")
        return image

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

            black_crop = self.black_mask[int(box_y):int(box_y+h), int(box_x):int(box_x+w)]
            if black_crop is not None:
                black_pixels = cv2.countNonZero(black_crop)
                black_percent = (black_pixels / ROI_area) * 100

                if (self.debugging == True):
                    cv2.imshow('Black percent mask', black_crop)
                    print("Black (%): ", black_percent)
            
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
                                black_percent,
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

            white_crop = self.white_mask[int(box_y):int(box_y+h), int(box_x):int(box_x+w)]
            if white_crop is not None:
                white_pixels = cv2.countNonZero(white_crop)
                white_percent = (white_pixels / ROI_area) * 100

                if (self.debugging == True):
                    cv2.imshow('White percent mask', white_crop)
                    print("White (%): ", white_percent)
            
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
                                white_percent,
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
                    if area > 920:    

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
        self.white_filtered = self.morphology_filter(self.get_mask(2))
        self.black_filtered = self.morphology_filter(self.get_mask(3))

        # self.red_concatenated_mask = self.red1_filtered+self.red2_filtered

        #self.combined_mask = self.yellow_filtered + self.blue_filtered + self.white_filtered + self.black_filtered

        self.yellow_cone_mask = self.yellow_filtered + self.white_filtered
        self.blue_cone_mask = self.blue_filtered + self.black_filtered 

        self.yellow_contours = self.blob_extraction(image, self.yellow_cone_mask, 'yellow')
        self.blue_contours = self.blob_extraction(image, self.blue_cone_mask, 'blue')

        #cv2.imshow("all blobs", self.combined_mask)

        if (self.debugging == True):
            if self.yellow_filtered is not None:
                cv2.imshow('Filtered yellow', self.yellow_filtered)
            if self.blue_filtered is not None:
                cv2.imshow('Filtered blue', self.blue_filtered)
            if self.white_filtered is not None:
                cv2.imshow('Filtered white', self.white_filtered)
            if self.black_filtered is not None:
                cv2.imshow('Filtered black', self.black_filtered)

        return

def process_images(instance, cap, img):
    # images = instance.get_all_images()

    # for i in images:
    instance.reset()
    image = instance.pre_processing(img)
    instance.thresholding(img)
    instance.get_objects(img)
    classified = instance.classify(img)

    cv2.imshow('Classifier', classified)

   
    k = cv2.waitKey(5) & 0xFF
    if k == 27: #esc
        cap.release()
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 


# Main function. Run this to classify. Change path in get_image_paths() to your path
def main():
    p = Pipeline()
    # p.load_images_from_folder()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, image = cap.read()
        process_images(p,cap,image)

    cap.release()
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    main()