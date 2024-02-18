import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys

# binary image using threshold
img = cv.imread(cv.samples.findFile("images/cat1.jpg"), cv.IMREAD_GRAYSCALE)
ret, thresh1 = cv.threshold(img, 27, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thresh3 = cv.threshold(img, 227, 255, cv.THRESH_BINARY)

images = [img, thresh1, thresh2, thresh3]
for i in range(4):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.xticks([]),plt.yticks([])
plt.show()

# conected components
inverted_image = cv.bitwise_not(thresh2)
num_labels, labels = cv.connectedComponents(inverted_image, connectivity=4)
print(num_labels)
label_hue = np.uint8(500*labels/np.max(labels))
print(label_hue)
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

labeled_img[label_hue == 0] = 0

plt.imshow(cv.cvtColor(labeled_img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Image after Component Labeling")
plt.show()