import numpy as np
from scipy.signal import convolve2d
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt

def depthwise_convolution(image, kernel):
    # Apply a kernel to each channel of the image
    output = np.zeros_like(image)
    for c in range(image.shape[2]):
        output[:, :, c] = convolve2d(image[:, :, c], kernel[:, :, c], mode='same', boundary='wrap')
    return output

def pointwise_convolution(image, kernel):
    # Apply a 1x1 kernel across the channels of the image
    output = np.zeros((image.shape[0], image.shape[1], kernel.shape[3]))
    for i in range(kernel.shape[3]):  # Iterate over the output channels
        for j in range(image.shape[2]):  # Iterate over the input channels
            output[:, :, i] += image[:, :, j] * kernel[0, 0, j, i]
    return output

# Load and preprocess the image
image = data.astronaut()
image = resize(image, (image.shape[0] // 2, image.shape[1] // 2), anti_aliasing=True)
image_gray = rgb2gray(image)
image_rgb = np.stack((image_gray,) * 3, axis=-1)

# Define a kernel for edge detection
kernel_edge = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel_edge = np.stack((kernel_edge,) * 3, axis=2)

# Define a kernel for pointwise convolution
kernel_pointwise = np.ones((1, 1, 3, 1))  # Combine channels into a single channel

# Apply the convolutions
convolved_depthwise = depthwise_convolution(image_rgb, kernel_edge)
convolved_pointwise = pointwise_convolution(convolved_depthwise, kernel_pointwise)

# Plotting
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(convolved_depthwise)
plt.title('Depthwise Convolved Image')
plt.axis('off')

plt.subplot(1, 3, 3)
# Squeeze is used to reduce the third dimension if it's 1
plt.imshow(convolved_pointwise.squeeze(), cmap='gray')
plt.title('Pointwise Convolved Image')
plt.axis('off')

plt.show()
