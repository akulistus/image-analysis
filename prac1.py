import matplotlib as plt
import numpy as np
import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("images/bones_1.jpg"))

if img is None:
    sys.exit("Could not read image.")

cv.imshow("Display windows", img)
k = cv.waitKey(0)

# if k == ord("s"):
#     cv.imwrite("new_img.png", img)

# pixel color
px = img[100, 100]
print(f"Color of pixel = {px}")

# modify pixel color
img[100, 100] = [0, 255, 0]
cv.imshow("Diss", img)

# get image properties
print(f"Number of rows, cols, channels {img.shape}")

# get total number of pixels
print(f"Total number of pixels {img.size}")

# get datatype 
print(f"Datatype {img.dtype}")

