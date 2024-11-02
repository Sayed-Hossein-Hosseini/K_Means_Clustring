from sys import argv
import numpy as np
from PIL import Image

def initialize_centroids(allPixels):
    choices = np.random.choice(range(len(allPixels)), size=K, replace=False)
    return [allPixels[matrix] for matrix in choices]

def k_means_clustring(allPixels):

    # cluster number
    K = int(argv[1])

    Z = initialize_centroids(allPixels)

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

            if j not in Z:
                # distance calculater
                distance = np.linalg.norm(allPixels[i] - Z[j])

                # best cluster
                if(distance < min_distance and j not in Z):
                    min_distance = distance
                    min_index = j

        G[min_index].append(i)

    return G

def update_centroids(allPixels, K, G):
    #Clear centroids Collection
    Z = [k for k in range(K)]

    for i in range(K):
        avreage = np.zeros((256, 1))
        for j in G[i]:
            avreage += allPixels[j]

        avrage /= len(G[i])
        Z[i] = average

    return Z

def main():
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

                # 16 * 16 convert 256 * 1
                pixel = np.array(cropped_image)
                shape = np.reshape(pixel, (256, 1))

                if not np.equal(np.linalg.norm(shape), 0):
                    allPixels.append(shape)

    Z, G = k_means_clustring(allPixels)


if __name__ == "__main__":
    main()