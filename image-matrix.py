import numpy as np
import cv2

image_path = 'car.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

image_matrix = np.array(image)

np.savetxt('image_matrix.txt', image_matrix, fmt='%d')

print("Image matrix saved to 'image_matrix.txt'")