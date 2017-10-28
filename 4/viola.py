import numpy as np

def get_integral_image(image):
    a = np.zeros((image.shape[0] + 1, image.shape[1] + 1))
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            a[i + 1, j + 1] = image[i, j] + a[i + 1, j] + a[i, j + 1] - a[i, j]
    return a

def calculate_sum(i_image, x, y, h, w):
    return i_image[h, w] - i_image[x - 1, w] - i_image[h, y - 1] + i_image[x - 1, y - 1]

def extract_features(image):
    integral_image = get_integral_image(image)
    n = image.shape[0]
    m = image.shape[1]
    features = []
    for i in xrange(1, n + 1):
        for j in xrange(1, m + 1):
            for h in xrange(6, 10):
                for w in xrange(6, 10):
                    if i + h - 1 < n + 1 and j + 2 * w - 1 < m + 1:
                        s1 = calculate_sum(integral_image, i, j,     i + h - 1, j + w - 1)
                        s2 = calculate_sum(integral_image, i, j + w, i + h - 1, j + 2 * w - 1)
                        features.append(s1 - s2)
                    if i + h - 1 < n + 1 and j + 3 * w - 1 < m + 1:
                        s1 = calculate_sum(integral_image, i, j,     i + h - 1, j + w - 1)
                        s2 = calculate_sum(integral_image, i, j + w, i + h - 1, j + 2 * w - 1)
                        s3 = calculate_sum(integral_image, i, j + 2 * w, i + h - 1, j + 3 * w - 1)
                        features.append(s1 - s2 + s3)
                    if i + 2 * h - 1 < n + 1 and j + w - 1 < m + 1:
                        s1 = calculate_sum(integral_image, i,     j, i + h - 1,     j + w - 1)
                        s2 = calculate_sum(integral_image, i + h, j, i + 2 * h - 1, j + w - 1)
                        features.append(s1 - s2)
                    if i + 3 * h - 1 < n + 1 and j + w - 1 < m + 1:
                        s1 = calculate_sum(integral_image, i,         j, i + h - 1,     j + w - 1)
                        s2 = calculate_sum(integral_image, i + h,     j, i + 2 * h - 1, j + w - 1)
                        s3 = calculate_sum(integral_image, i + 2 * h, j, i + 3 * h - 1, j + w - 1)
                        features.append(s1 - s2 + s3)
                    if i + 2 * h - 1 < n + 1 and j + 2 * w - 1 < m + 1:
                        s1 = calculate_sum(integral_image, i,     j,     i + h - 1,     j + w - 1)
                        s2 = calculate_sum(integral_image, i + h, j,     i + 2 * h - 1, j + w - 1)
                        s3 = calculate_sum(integral_image, i,     j + w, i + h - 1,     j + 2 * w - 1)
                        s4 = calculate_sum(integral_image, i + h, j + w, i + 2 * h - 1, j + 2 * w - 1)
                        features.append(s1 - s2 - s3 + s4)
    return features