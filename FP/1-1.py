import numpy as np
import cv2
import matplotlib.pyplot as plt

#image = cv2.imread("Dataset/1/1-1/parrots.jpg")
image = cv2.imread("Dataset/1/1-1/IMG_20201226_004745_599.jpg")

#convert image to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#convert imaage to (W*H,3)
pixel_values = image.reshape((-1, 3))

pixel_values = np.float32(pixel_values)

print(pixel_values.shape)

#criteria for K-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
k_list = [2,3,4,5,6,10,15,20]

#apply k-means and show image for each k
for k in k_list:
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    plt.imshow(segmented_image)
    plt.title(f'k={k}')
    plt.show()

