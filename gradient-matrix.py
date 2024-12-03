import numpy as np
import cv2

image_path = 'car.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

image = cv2.flip(image, 0) # fix flip

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

gradient_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
gradient_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)

gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

gradient_magnitude = np.uint8(gradient_magnitude)

np.savetxt('gradient_matrix.txt', gradient_magnitude, fmt='%d')

print("Gradient matrix saved to 'gradient_matrix.txt'")