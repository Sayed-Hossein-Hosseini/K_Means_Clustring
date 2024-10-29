import random
from sys import argv
import numpy as np
from PIL import Image

# All pixels cropped image
allPixels = []

for picture in range(5):
    # Open images
    image_path = "D://University//Term 7//Linear Algebra//Project//k-means-clustring//DataSet//usps_" + str(picture + 1) + ".jpg"
    image = Image.open(image_path)
    # find row and coulmn
    row_image, coulmn_image = np.array(image).shape
    row_image /= 16
    coulmn_image /= 16

    for x in range(int(coulmn_image)):
        for y in range(int(row_image)):
            # crop image pixels 16 * 16
            crop_area = (x * 16, y * 16,(x + 1) * 16,(y + 1) * 16)
            cropped_image = image.crop(crop_area)
            cropped_image.save("D://University//Term 7//Linear Algebra//Project//k-means-clustring//DataSet//Cropped_Image//usps_cropped_" + str(picture + 1) + "_" + str(x) + "_" + str(y) + ".jpg")
            # 16 * 16 convert 256 * 1
            pixel = np.array(cropped_image)
            shape = np.reshape(pixel, (256, 1))
            allPixels.append(shape)

# def k_means_clustring(allPixels):
#     # cluster number
#     k = int(argv[1])
#     # Representatives Intioialization
#     Z = [np.random.sample(allPixels, k)]
#     # Clusters Intioialization
#     G = [[None]] * k


# def choose_cluster(Z, G):


# def choose_represenative():
#
#
# def j_cluster():