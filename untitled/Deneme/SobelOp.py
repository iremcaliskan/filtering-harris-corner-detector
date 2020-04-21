import cv2
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy import ndimage as ndi
from skimage.feature import corner_harris, corner_peaks

# Load the image, convert it to grayscale, and show it
image = cv2.imread("gradients1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Greyscale", image)
cv2.imwrite("greyscale-img.png", image)

# Compute the Laplacian of the image
lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian", lap)
cv2.imwrite("laplacian-img.png", lap)

cv2.waitKey(0)

# Compute gradients along the X and Y axis, respectively
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

# The sobelX and sobelY images are now of the floating
# point data type -- we need to take care when converting
# back to an 8-bit unsigned integer that we do not miss
# any images due to clipping values outside the range
# of [0, 255]. First, we take the absolute value of the
# graident magnitude images, THEN we convert them back
# to 8-bit unsigned integers
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

# We can combine our Sobel gradient images using our
# bitwise OR
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# Show our Sobel images
cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.imwrite("sobelX-img.png", sobelX)
cv2.imwrite("sobelY-img.png", sobelY)
cv2.imwrite("sobelcombined-img.png", sobelCombined)

cv2.waitKey(0)

# Load the image, convert it to grayscale, and blur it
# slightly to remove high frequency edges that we aren't
# interested in
image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Blurred", image)
cv2.imwrite("blurred.png", image)

# When performing Canny edge detection we need two values
# for hysteresis: threshold1 and threshold2. Any gradient
# value larger than threshold2 are considered to be an
# edge. Any value below threshold1 are considered not to
# ben an edge. Values in between threshold1 and threshold2
# are either classified as edges or non-edges based on how
# the intensities are "connected". In this case, any gradient
# values below 30 are considered non-edges whereas any value
# above 150 are considered edges.
canny = cv2.Canny(image, 30, 150)
cv2.imshow("Canny", canny)
cv2.imwrite("canny-img.png", canny)
cv2.waitKey(0)

img = imread('gradients1.jpg')
imggray = rgb2gray(img)

plt.imshow(imggray, cmap="gray"),plt.title('Remember Original Form Again')
plt.axis("off")
plt.show()

def gradient_x(imggray):
    ##Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return sig.convolve2d(imggray, kernel_x, mode='same')
def gradient_y(imggray):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sig.convolve2d(imggray, kernel_y, mode='same')

I_x = gradient_x(imggray)
I_y = gradient_y(imggray)

Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
Iyy = ndi.gaussian_filter(I_y**2, sigma=1)

k = 0.05

# determinant
detA = Ixx * Iyy - Ixy ** 2
# trace
traceA = Ixx + Iyy

harris_response = detA - k * traceA ** 2

# height, width = imggray.shape
# harris_response = []
# window_size = 6
# offset = int(window_size/2)
# for y in range(offset, height-offset):
#     for x in range(offset, width-offset):
#         Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
#         Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
#         Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])

#         #Find determinant and trace, use to get corner response
#         det = (Sxx * Syy) - (Sxy**2)
#         trace = Sxx + Syy
#         r = det - k*(trace**2)

#         harris_response.append(r)

img_copy_for_corners = np.copy(img)
img_copy_for_edges = np.copy(img)

for rowindex, response in enumerate(harris_response):
    for colindex, r in enumerate(response):
        if r > 0:
            # this is a corner
            img_copy_for_corners[rowindex, colindex] = [255, 0, 0]
        elif r < 0:
            # this is an edge
            img_copy_for_edges[rowindex, colindex] = [0, 255, 0]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
ax[0].set_title("corners found")
ax[0].imshow(img_copy_for_corners)
ax[1].set_title("edges found")
ax[1].imshow(img_copy_for_edges)
plt.show()
cv2.imwrite("img_copy_for_edges.png", img_copy_for_edges)
cv2.imwrite("img_copy_for_corners.png", img_copy_for_corners)



corners = corner_peaks(harris_response)
fig, ax = plt.subplots()
ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
ax.plot(corners[:, 1], corners[:, 0], '.r', markersize=3)
