# K-Means Clustering on MNIST and Fashion MNIST

## Project Overview

This project implements **K-Means clustering** on the **MNIST** and **Fashion MNIST** datasets using Python. The K-Means algorithm is an unsupervised machine learning technique used to partition data into clusters. This project clusters images of handwritten digits (MNIST) and fashion items (Fashion MNIST) to group similar data points together.

## K-Means Clustering Algorithm

The **K-Means clustering algorithm** is based on iterative optimization to form clusters by grouping similar data points. The algorithm works as follows:

1. **Initialization**: Randomly select `K` initial representatives (`Z`) for each cluster.
2. **Assignment Step**: For each data point in the dataset (`allPixel`), find the closest representative in `Z` and assign it to the corresponding cluster `G`.
3. **Update Step**: Recalculate the representatives `Z` as the mean of all data points in each cluster `G`.
4. **Repeat**: Repeat the assignment and update steps until the representatives `Z` stabilize or a maximum iteration limit is reached.

### Key Parameters

In this implementation, we use the following parameters:

- **`K` (Number of Clusters)**: This is the number of clusters or groups we want to create. In this project, we set `K` as desired to check all kinds of situations, in such a way that for the <b>MNIST</b> dataset, we set the value of `K` to `3`, `4`, `5`, `6`, and `7`, and for <b>Fashion MNIST</b>, due to the long processing time, we set the same number of main categories as `10`. 
  
- **`allPixel` (Data Points)**: This is the dataset containing all pixel values of the images. Each image is flattened into a vector, representing each pixel as a feature, so `allPixel` becomes a collection of vectors (data points) for clustering.

- **`Z` (Centroids / Representatives)**: These are the centroids or "representative points" for each cluster, initialized randomly from `allPixel`. The K-Means algorithm updates `Z` in each iteration by calculating the mean position of all data points in each cluster. These centroids represent the "center" of each cluster.

- **`G` (Clusters)**: `G` is the set of clusters, where each cluster contains the data points assigned to the same representative (`Z`). During the assignment step, each point in `allPixel` is assigned to the closest representative in `Z`, forming the clusters `G`.

### Algorithm Flow

1. **Initialize `Z`**: Randomly select `K` points from `allPixel` to serve as the initial representatives.
2. **Cluster Assignment**:
   - For each data point in `allPixel`, find the nearest representative in `Z` based on a distance metric (e.g., Euclidean distance).
   - Assign each point to the corresponding cluster in `G`.
3. **Centroid Update**:
   - For each cluster in `G`, recalculate the representative `Z` by averaging the points within that cluster.
4. **Convergence Check**: Repeat the steps above until the representatives in `Z` do not change significantly or until a maximum number of iterations is reached.

### Algorithm Execution Instructions

To run the algorithm, write the following command in the terminal and set a value according to your liking for the number of clusters:<br>
(K is the number of cluster)
```bash
python main.py K
```

## Datasets

### MNIST Dataset
<p align="center">
  <img src="https://i.ytimg.com/vi/0QI3xgXuB-Q/hqdefault.jpg" alt="MNIST Clustering Example" />
</p><br>
The MNIST dataset consists of grayscale images of handwritten digits (1-5), with 6,000 images. Each image is 16x16 pixels, which are flattened to vectors of size 256 for clustering.

### Fashion MNIST Dataset
<p align="center">
  <img src="https://production-media.paperswithcode.com/datasets/Fashion-MNIST-0000000040-4a13281a_m8bp4wm.jpg" alt="MNIST Clustering Example" />
</p><br>
The Fashion MNIST dataset includes grayscale images of various clothing items (e.g., T-shirts, shoes, and bags). Like MNIST, it contains 70,000 images, each 28x28 pixels in size.

## Project Structure

- **K_Means_Clustring Directory**: The main Python script that loads the dataset, applies K-Means clustering, and displays the results.
- **Dataset Directory**: MNIST and Fashion MNIST datasets are located in it for processing.
- **Hossein Directory**: The data after processing are datasets, which include representatives of the clusters as well as the values ​​assigned to each cluster.

## Requirements

- Python Language
- kagglehub (For downloading datasets)
- Numpy (For work with arrays)
- Pillow (For process images)
- sys (For work with the terminal)
- csv (For work with the Dataset)

You can install the required libraries with the following command:

```bash
pip install library-name
```

## Authors

- Sayyed Hossein Hosseini Dolat Abadi

## License

This project is licensed under the MIT License.
