import cv2
import numpy as np

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold(image):
    gray = to_grayscale(image)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def dilation(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erosion(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def median_blur(image):
    return cv2.medianBlur(image, 5)

def canny_edge(image):
    return cv2.Canny(image, 100, 200)

def sobel_edge(image):
    gray = to_grayscale(image)
    return cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)

def rotate_image(image):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def resize_image(image, scale=0.5):
    return cv2.resize(image, None, fx=scale, fy=scale)
