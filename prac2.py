import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

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

img = cv.imread(cv.samples.findFile("images/forComp.jpg"), 0)
img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
# conected components
# тут посмотреть
num_labels, labels = cv.connectedComponents(img, connectivity=4)
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

labeled_img[label_hue == 0] = 0

plt.imshow(cv.cvtColor(labeled_img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Image after Component Labeling")
plt.show()

num_labels, labels = cv.connectedComponents(img, connectivity=8)
label_hue = np.uint8(178*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

labeled_img[label_hue == 0] = 0

plt.imshow(cv.cvtColor(labeled_img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Image after Component Labeling")
plt.show()

# histogramm equalization
# numpy
img = cv.imread(cv.samples.findFile("images/landscape.jpg"), cv.IMREAD_GRAYSCALE)
hist,bins = np.histogram(img.flatten(),256, [0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

plt.plot(cdf_m, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

res = np.hstack((img,img2))
cv.imshow("OpenCV equalization", res)
cv.waitKey(0)

# opencv
plt.show()
equ = cv.equalizeHist(img)
res = np.hstack((img,equ))
cv.imshow("OpenCV equalization", res)
cv.waitKey(0)