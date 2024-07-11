import cv2
import numpy as np
import imutils
import easyocr

# Load the image
img = cv2.imread("images/5.jpg")
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(grey,11,17,17)
edge=cv2.Canny(bfilter,30,100)
edgeCountour= edge.copy()

keypoints = cv2.findContours(edgeCountour, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
countours = imutils.grab_contours(keypoints)
countours = sorted(countours,key=cv2.contourArea,reverse=True)[:10]
location = None

for contour in countours:
    approx = cv2.approxPolyDP(contour,10,True)
    if len (approx)== 4 :
        location = approx
        break

mask = np.zeros(grey.shape,np.uint8)
new_image =  cv2.drawContours(mask,[location],0,255,-1)
new_image = cv2.bitwise_and(img,img, mask=mask)

(x, y, w, h) = cv2.boundingRect(location)
cropped_image = img[y:y+h, x:x+w]

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

if result:
    text = result[0][-2]
    text_position = (x, y - 10)  # Set the position above the bounding box for better visibility

    # Draw the text on the original image
    res = cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    res = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
else:
    print("No text detected")
    # Display the results
    cv2.imshow("Cropped Number Plate", cropped_image)
if result:
    cv2.imshow("Image with Text", cv2.cvtColor(res, cv2.COLOR_BGR2RGB)) 
    print(result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Number plate not found.")