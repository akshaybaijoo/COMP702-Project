import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

img = cv2.imread('C.jpeg')

cv2.imshow('Original', img)
cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

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

def calculate_glcm_features(image, distances, angles, properties):
    # Convert the image to grayscale if necessary
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the image to 8-bit grayscale
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Calculate the GLCM
    glcm = graycomatrix(image, distances, angles, symmetric=True, normed=True)

    # Calculate the specified GLCM properties
    features = []
    for prop in properties:
        feature = graycoprops(glcm, prop)
        features.extend(feature.flatten())

    return features

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

        distances = [1]  # Distance between pixels
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Directions of co-occurrence
        properties = ['contrast', 'energy', 'homogeneity']  # GLCM properties to compute

        # Calculate GLCM features
        glcm_features = calculate_glcm_features(coin, distances, angles, properties)

        # Print the extracted GLCM features
        for feature_name, feature_value in zip(properties, glcm_features):
            print(f'{feature_name}: {feature_value}')

coin_labels = ["5-1.jpeg", "5-2.jpeg", "5-3.jpeg", "5-4.jpeg", "10c_O 1.png", "10c_R1.png", "10c.png", "20c_O 1.png", "20c_R1.png", "20c.png", "50c_O 1.png", "50c_R1.png", "50c.png",  "R1_O 1.png", "R1_R1.png", "R1.png",  "R2_O 1.png", "R2_R1.png", "R2.png",  "R5_O 1.png", "R5_RA.png", "R5.png", "C.jpeg", "Coins.jpeg"]

X_train, X_test, y_train, y_test = train_test_split(glcm_features, coin_labels, test_size=0.2, random_state=42)


svm_classifier = SVC()

svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

