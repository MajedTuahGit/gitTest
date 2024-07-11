import cv2
import numpy as np
import imutils
import easyocr

# Load the image
img = cv2.imread("images/1.jpg")

# Check if the image was successfully loaded
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Detect edges
edge = cv2.Canny(bfilter, 30, 200)

    # Find contours
keypoints = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)

    # Sort contours based on area and take the largest ones
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

if location is not None:
    # Create a mask for the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)

    # Bitwise AND the mask and original image
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Crop the number plate region
    (x, y, w, h) = cv2.boundingRect(location)
    cropped_image = img[y:y+h, x:x+w]

    # Display the results
    cv2.imshow("Original Image", img)
    cv2.imshow("Grayscale Image", gray)
    cv2.imshow("Canny Edge Detection", edge)
    cv2.imshow("Masked Image", new_image)
    cv2.imshow("Cropped Number Plate", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Number plate not found.")
