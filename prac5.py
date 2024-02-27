import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# gamma correction
def gammaCorrection(img, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv.LUT(img, lookUpTable)

    return res

img = cv.imread(cv.samples.findFile("images/dark.jpg"))
gamma_03 = cv.cvtColor(gammaCorrection(img, 0.3), cv.COLOR_BGR2RGB)
gamma_07 = cv.cvtColor(gammaCorrection(img, 0.7), cv.COLOR_BGR2RGB)
gamma_13 = cv.cvtColor(gammaCorrection(img, 1.3), cv.COLOR_BGR2RGB)
RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.subplot(3,2,1)
plt.imshow(RGB_img)
plt.xticks([]), plt.yticks([])
plt.title("Original")
plt.subplot(3,2,2)
plt.imshow(gamma_03)
plt.xticks([]), plt.yticks([])
plt.title("Gamma_03")
plt.subplot(3,2,3)
plt.imshow(RGB_img)
plt.xticks([]), plt.yticks([])
plt.title("Original")
plt.subplot(3,2,4)
plt.imshow(gamma_07)
plt.xticks([]), plt.yticks([])
plt.title("Gamma_07")
plt.subplot(3,2,5)
plt.imshow(RGB_img)
plt.xticks([]), plt.yticks([])
plt.title("Original")
plt.subplot(3,2,6)
plt.imshow(gamma_13)
plt.xticks([]), plt.yticks([])
plt.title("Gamma_13")
plt.show()

# 

dogs = cv.imread(cv.samples.findFile("images/dogs.jpg"), cv.IMREAD_GRAYSCALE)
dog = cv.imread(cv.samples.findFile("images/dog.jpg"), cv.IMREAD_GRAYSCALE)

w, h = dog.shape[::-1]

res = cv.matchTemplate(dogs, dog, cv.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(dogs,top_left, bottom_right, 255, 2)
plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dogs,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(cv.TM_CCORR_NORMED)
plt.show()