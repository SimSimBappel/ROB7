import glob
import cv2
import numpy as np

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob("lec3/exercise_materials/Test016/*.tif")]

current_image = 0
quit = False
alpha = 0.5

background_subtractor = cv2.createBackgroundSubtractorMOG2()

while not quit:
    test_image = images[current_image]
    foreground_mask = background_subtractor.apply(test_image)
    foreground_mask = foreground_mask.astype(np.uint8)

    output_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
    empty_image = np.zeros(foreground_mask.shape, foreground_mask.dtype)
    foreground_mask_red = np.stack([empty_image, empty_image, foreground_mask], axis=2)
    output_image = cv2.addWeighted(output_image, alpha, foreground_mask_red, 1.0 - alpha, 0)

    cv2.imshow("Output", output_image)

    key = cv2.waitKey(0)
    if key == 113 or key == 27: # q or Esc
        break
    if key == 83: # Right arrow
        current_image = min(current_image + 1, len(images)-1)
    if key == 81:  # Left arrow
        current_image = max(current_image - 1, 0)