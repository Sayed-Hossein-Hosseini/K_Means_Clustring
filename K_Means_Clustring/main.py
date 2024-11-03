from sys import argv

import kagglehub
import numpy as np
from PIL import Image


def download_Fashion_MNIST_Dataset():
    # Download latest version
    path = kagglehub.dataset_download("zalando-research/fashionmnist")

    print("Path to dataset files:", path)


def initialize_centroids(allPixels, K):
    choices = np.random.choice(range(len(allPixels)), size=K, replace=False)
    return [allPixels[matrix] for matrix in choices]


def k_means_clustring(allPixels):
    # cluster number
    K = int(argv[1])

    Z = initialize_centroids(allPixels, K)

    for _ in range(100):
        G = update_cluster(allPixels, K, Z)
        Z = update_centroids(allPixels, K, G)

    return Z, G


def update_cluster(allPixels, K, Z):
    # Clear Clusters Collection
    G = [[] for _ in range(K)]
    number_cropped_image = len(allPixels)

    for i in range(number_cropped_image):
        min_distance = int(1e100)
        min_index = 0

        for j in range(K):

            # if allPixels[j] not in Z:
            # distance calculater
            distance = np.linalg.norm(allPixels[i] - Z[j])

            # best cluster
            if (distance < min_distance):
                min_distance = distance
                min_index = j

        G[min_index].append(i)

    return G


def update_centroids(allPixels, K, G):
    # Clear centroids Collection
    Z = [k for k in range(K)]

    for i in range(K):
        average = np.zeros((256, 1))
        for j in G[i]:
            average += allPixels[j]

        average /= len(G[i])
        Z[i] = average

    return Z


def main_MNIST():
    # All pixels cropped image
    allPixels = []
    for picture in range(5):
        # Open images
        image_path = "D://University//Term 7//Linear Algebra//Project//k-means-clustring//DataSet//MNIST//usps_" + str(
            picture + 1) + ".jpg"
        image = Image.open(image_path)
        # find row and coulmn
        row_image, coulmn_image = np.array(image).shape
        row_image /= 16
        coulmn_image /= 16

        for x in range(int(coulmn_image)):
            for y in range(int(row_image)):
                # crop image pixels 16 * 16
                crop_area = (x * 16, y * 16, (x + 1) * 16, (y + 1) * 16)
                cropped_image = image.crop(crop_area)

                # 16 * 16 convert 256 * 1
                pixel = np.array(cropped_image)
                shape = np.reshape(pixel, (256, 1))

                if not np.equal(np.linalg.norm(shape), 0):
                    allPixels.append(shape)

    Z, G = k_means_clustring(allPixels)

    for i in range(len(Z)):
        centroid = Z[i]
        centroid_matrix = centroid.reshape((16, 16))
        image_data = np.uint8(centroid_matrix)

        image = Image.fromarray(image_data)

        image.save(
            f"D:\\University\\Term 7\\Linear Algebra\\Project\\k-means-clustring\\Hossein\\MNIST\\K = {len(Z)}\\Centroids\\centroid_{i + 1}.jpg")
        image.show()

        for j in range(len(G[i])):
            element = allPixels[G[i][j]]
            element_matrix = element.reshape((16, 16))
            image_data = np.uint8(element_matrix)

            image = Image.fromarray(image_data)

            image.save(
                f"D:\\University\\Term 7\\Linear Algebra\\Project\\k-means-clustring\\Hossein\\MNIST\\K = {len(Z)}\\Clusters\\{i + 1}\\cluster_{j + 1}.jpg")
            image.show()


if __name__ == "__main__":
    main_MNIST()
