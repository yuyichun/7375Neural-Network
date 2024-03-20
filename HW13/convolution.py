import numpy as np


def convolve(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output_dimension = np.array(image.shape) - np.array(kernel.shape) + 1
    # Initialize the output feature map
    new_image = np.zeros(output_dimension)
    for x in range(output_dimension[0]):
        for y in range(output_dimension[1]):
            # Element-wise multiplication and summation
            new_image[x, y] = np.sum(
                image[x:x+kernel.shape[0], y:y+kernel.shape[1]] * kernel)
    return new_image


# Define a 6x6 image
image = np.array([
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0]
])

# Define a 3x3 filter
kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

# Apply the convolution operation
convolved_image = convolve(image, kernel)
print(convolved_image)
