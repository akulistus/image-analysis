import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(cv.samples.findFile("images/cat1.jpg"), cv.IMREAD_GRAYSCALE)

# apply noise and denoise
imp_noise=np.zeros(img.shape, dtype=np.uint8)
cv.randu(imp_noise, 0, 255)
imp_noise=cv.threshold(imp_noise, 200, 255, cv.THRESH_BINARY)[1]

in_img=cv.add(img,imp_noise)
denoised_img = cv.medianBlur(in_img, 9)

plt.subplot(1,4,1)
plt.imshow(img,cmap='gray')
plt.axis("off")
plt.title("Original")
plt.subplot(1,4,2)
plt.imshow(imp_noise,cmap='gray')
plt.axis("off")
plt.title("Impulse Noise")
plt.subplot(1,4,3)
plt.imshow(in_img,cmap='gray')
plt.axis("off")
plt.title("Combined")
plt.subplot(1,4,4)
plt.imshow(denoised_img,cmap='gray')
plt.axis("off")
plt.title("Denoised")

plt.show()

# canny

edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# median filtering
# поменять на N-порядок
def medinaFilter(img, ksize, N):
    final = np.zeros(img.shape)

    size = ksize//2
    for y in range(size,img.shape[0]-size):
        for x in range(size,img.shape[1]-size):
            kernel = img[y-size:y+size+1, x-size:x+size+1].copy()
            kernel = np.sort(np.array(kernel).flatten())
            final[y, x] = kernel[N]

    return final

denoise_3 = medinaFilter(in_img, 3, 1)
denoise_5 = medinaFilter(in_img, 5, 1)
denoise_7 = medinaFilter(in_img, 7, 1)

# make image bigger
plt.subplot(3,2,1),plt.imshow(in_img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(denoise_3,cmap = 'gray')
plt.title('denoise_3 Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,3),plt.imshow(in_img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,4),plt.imshow(denoise_5,cmap = 'gray')
plt.title('denoise_5 Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(in_img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(denoise_7,cmap = 'gray')
plt.title('denoise_7 Image'), plt.xticks([]), plt.yticks([])
plt.show()