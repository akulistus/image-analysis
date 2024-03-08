import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread(cv.samples.findFile('images/dogs.jpg'), cv.IMREAD_GRAYSCALE)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# highpass filter

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

mask = np.ones((rows, cols), np.uint8)
r = 80
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0

fshift = fshift * mask
magnitude_spectrum = 20*np.log(np.abs(fshift))
unshihfted = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(unshihfted)
img_back = np.real(img_back)

plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
plt.title('HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# lowpass filter

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

mask = np.zeros((rows, cols), np.uint8)
r = 80
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

fshift = fshift * mask
magnitude_spectrum = 20*np.log(np.abs(fshift))
unshihfted = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(unshihfted)
img_back = np.real(img_back)

plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
plt.title('LPF'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# bandpass filter

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

mask = np.zeros((rows, cols), np.uint8)
r_in = 10
r_out = 50
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
mask[mask_area] = 1

fshift = fshift * mask
magnitude_spectrum = 20*np.log(np.abs(fshift))
unshihfted = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(unshihfted)
img_back = np.real(img_back)

plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
plt.title('BPF'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()