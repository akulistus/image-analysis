import matplotlib.pyplot as plt
import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("images/cat.jpg"))

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
k = cv.waitKey(0)

# get image properties
print(f"Number of rows, cols, channels {img.shape}")

# get total number of pixels
print(f"Total number of pixels {img.size}")

# get datatype 
print(f"Datatype {img.dtype}")

# zoom/shrink image 2x
res_shrink = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
res_zoom = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

cv.imshow("0.5x", res_shrink)
k = cv.waitKey(0)

cv.imshow("2x", res_zoom)
k = cv.waitKey(0)

# zoom 2x whith nearest, linear, cubic
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST)
cv.imshow("x2_NEAREST", res)
k = cv.waitKey(0)

res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
cv.imshow("x2_LINEAR", res)
k = cv.waitKey(0)

res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
cv.imshow("x2_CUBIC", res)
k = cv.waitKey(0)

# grayscale image
img_gray = cv.imread("images/cat.jpg", cv.IMREAD_GRAYSCALE)
plt.hist(img_gray.ravel(), 256, [0, 256])
plt.show()

# color image
for i in range(3):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr)
    plt.xlim([0, 256])
plt.show()