from sys import argv
import numpy as np
from PIL import Image

def k_means_clustring(allPixels):

    # cluster number
    K = int(argv[1])
    # Representatives Intioialization
    choices = np.random.choice(range(len(allPixels)), size=K, replace=False)
    Z = [allPixels[matrix] for matrix in choices]
    # Clusters Intioialization
    G = [[] for _ in range(K)]
    G = choose_cluster(allPixels, K, Z, G)
    # Z = choose_represenative(Z, G)
    # flag = j_cluster(Z, G)


def choose_cluster(allPixels, K, Z, G):
    # Clear Clusters Collection
    G = [[] for _ in range(K)]
    number_cropped_image = len(allPixels)
    for i in range(number_cropped_image):
        min_distance = 0
        min_index = 0
        for j in range(K):
            # distance calculater
            if(j == 0):
                min_distance = np.linalg.norm(allPixels[i] - Z[j])
                distance = min_distance

            else:
                distance = np.linalg.norm(allPixels[i] - Z[j])

            # best cluster
            if(distance < min_distance):
                min_distance = distance
                min_index = j

        G[min_index].append(allPixels[i])

    return G

def j_cluster(Z, G):
    pass

def choose_represenative(Z, G):
    pass

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
                # cropped_image.save("D://University//Term 7//Linear Algebra//Project//k-means-clustring//DataSet//Cropped_Image//usps_cropped_" + str(picture + 1) + "_" + str(x) + "_" + str(y) + ".jpg")
                # 16 * 16 convert 256 * 1
                pixel = np.array(cropped_image)
                shape = np.reshape(pixel, (256, 1))
                allPixels.append(shape)

    k_means_clustring(allPixels)

if __name__ == "__main__":
    main()