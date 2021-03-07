# very detail source https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html
# Not dive deep yet, just copy paste
import numpy as np
import torch
import math


def linear_resize(in_array, size):
    """
    `in_array` is the input array.
    `size` is the desired size.
    """
    ratio = (len(in_array) - 1) / (size - 1)
    out_array = []

    for i in range(size):
        low = math.floor(ratio * i)
        high = math.ceil(ratio * i)
        weight = ratio * i - low

        a = in_array[low]
        b = in_array[high]

        out_array.append(a * (1 - weight) + b * weight)

    return out_array


def bilinear_resize(image, height, width):
    """
    `image` is a 2-D numpy array
    `height` and `width` are the desired spatial dimension of the new 2-D array.
    """
    img_height, img_width = image.shape[:2]

    resized = np.empty([height, width])

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    for i in range(height):
        for j in range(width):

            x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
            x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = image[y_l, x_l]
            b = image[y_l, x_h]
            c = image[y_h, x_l]
            d = image[y_h, x_h]

            pixel = (
                a * (1 - x_weight) * (1 - y_weight)
                + b * x_weight * (1 - y_weight)
                + c * y_weight * (1 - x_weight)
                + d * x_weight * y_weight
            )

            resized[i][j] = pixel

    return resized

def bilinear_resize_vectorized(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  img_height, img_width = image.shape

  image = image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_l = np.floor(x_ratio * x).astype('int32')
  y_l = np.floor(y_ratio * y).astype('int32')

  x_h = np.ceil(x_ratio * x).astype('int32')
  y_h = np.ceil(y_ratio * y).astype('int32')

  x_weight = (x_ratio * x) - x_l
  y_weight = (y_ratio * y) - y_l

  a = image[y_l * img_width + x_l]
  b = image[y_l * img_width + x_h]
  c = image[y_h * img_width + x_l]
  d = image[y_h * img_width + x_h]

  resized = a * (1 - x_weight) * (1 - y_weight) + \
            b * x_weight * (1 - y_weight) + \
            c * y_weight * (1 - x_weight) + \
            d * x_weight * y_weight

  return resized.reshape(height, width)

def main():
    np.array([[114., 195., 254., 217.,  33., 160.],
        [110.,  91., 184., 143., 190., 124.],
        [212., 163., 245.,  39.,  83., 188.],
        [ 23., 206.,  62.,   7.,   5., 206.],
        [152., 177., 118., 155., 245.,  41.]], dtype=np.float)
    pass


if __name__ == "__main__":
    main()
