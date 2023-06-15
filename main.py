import cv2
import numpy as np

img = cv2.imread('C.jpeg')

cv2.imshow('Original', img)
cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', img_gray)
cv2.waitKey(0)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
cv2.imshow('Gaussian Blur', img_blur)
cv2.waitKey(0)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()

equ = cv2.equalizeHist(edges)

circles = cv2.HoughCircles(equ, cv2.HOUGH_GRADIENT, 1, 50, param1=200, param2=10, minRadius=5, maxRadius=30)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(equ, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(equ, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('Hough Circles', equ)
cv2.waitKey(0)

if circles is not None:
    # Convert the coordinates and radii of the circles to integers
    circles = np.round(circles[0, :]).astype(int)

    for (x, y, r) in circles:
        # Create a mask for each coin by drawing a filled circle
        mask = np.zeros_like(equ, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Resize the mask to match the image size
        mask = cv2.resize(mask, (equ.shape[1], equ.shape[0]))

        # Apply the mask to the original image
        coin = cv2.bitwise_and(equ, equ, mask=mask)

        # Display the segmented coin region
        cv2.imshow('Coin', coin)
        cv2.waitKey(0)