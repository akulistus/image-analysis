import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(cv.samples.findFile("images/puzzle.jpg"), cv.IMREAD_GRAYSCALE)

# vertical and horisontal using sobel
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
# vertical and horisontal using laplacian
laplacian = cv.Laplacian(img, cv.CV_64F)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

# blur img using blur, boxFilter, gussian
# найти норм картинку!!!!!!!!!!!!!!!!
img = cv.imread(cv.samples.findFile("images/frog.jpg"))
blur = cv.blur(img, ksize=(5,5))
boxFilter = cv.boxFilter(img, cv.CV_64F, ksize=(5,5))
gaussian = cv.GaussianBlur(img, (5,5), 0)

plt.subplot(3,2,1),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(blur),plt.title('Blurred(cv.blur)')
plt.xticks([]), plt.yticks([])
plt.subplot(3,2,3),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(3,2,4),plt.imshow(boxFilter),plt.title('Blurred(cv.boxFilter)')
plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(gaussian),plt.title('Blurred(cv.gussian)')
plt.xticks([]), plt.yticks([])
plt.show()

# use filter2D

kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])

dst = cv.filter2D(img, -1, kernel)

plt.subplot(1,2,1),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(dst),plt.title('Blurred(cv.filter2D)')
plt.xticks([]), plt.yticks([])
plt.show()

