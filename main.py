import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
import numpy as np
import scipy


def manual_region_based_segmentation(image_path, threshold_value):
    # Load the image using matplotlib
    img = plt.imread(image_path)
    gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale

    # Manual thresholding
    thresh_img = np.where(gray_img > threshold_value, 255, 0)

    # Display images
    titles = ['Original Image', 'Manual Thresholded Image']
    images = [gray_img, thresh_img]

    for i in range(2):
        plt.subplot(2, 1, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


def region_based_segmentation(image_path, threshold_value):
    # Read the image
    img = cv2.imread(image_path, 0)

    # Apply global thresholding
    ret, thresh_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    # Display the original and thresholded images
    titles = ['Original Image', 'Thresholded Image']
    images = [img, thresh_img]

    for i in range(2):
        plt.subplot(2, 1, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


def manual_sobel_operator(image):
    # Define Sobel operators
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Manually convolve the image with Sobel operators
    Gx = scipy.signal.convolve2d(image, sobel_x, mode='same', boundary='symm')
    Gy = scipy.signal.convolve2d(image, sobel_y, mode='same', boundary='symm')

    # Calculate the magnitude of the gradient
    G = np.hypot(Gx, Gy)
    G = G / G.max() * 255  # Normalize to 0-255 scale

    return G


def manual_edge_detection_segmentation(image_path):
    img = plt.imread(image_path)
    gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale

    edges = manual_sobel_operator(gray_img)

    plt.subplot(121), plt.imshow(gray_img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Manual Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


def edge_detection_segmentation(image_path):
    img = cv2.imread(image_path, 0)
    edges = cv2.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


def manual_clustering_based_segmentation(image_path, n_clusters):
    img = plt.imread(image_path)
    pixel_values = img.reshape((-1, 3))

    # Convert pixel values to float32
    pixel_values = np.float32(pixel_values)

    # Perform k-means clustering
    centers, _ = scipy.cluster.vq.kmeans(pixel_values, n_clusters)

    # Assign each pixel to the closest cluster center
    labels = np.argmin(np.linalg.norm(pixel_values[:, None] - centers, axis=-1), axis=-1)

    # Reconstruct the segmented image
    segmented_image = centers[labels].reshape(img.shape)

    plt.imshow(segmented_image.astype(np.uint8))
    plt.show()


def clustering_based_segmentation(image_path, n_clusters):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)

    plt.imshow(segmented_image.astype(np.uint8))
    plt.show()


def main():
    image_path = 'example.jpg'

    # Region-based Segmentation
    print("Region-based Segmentation:")
    # region_based_segmentation(image_path, threshold_value=127)
    manual_region_based_segmentation(image_path, threshold_value=127)

    # Edge Detection Segmentation
    print("\nEdge Detection Segmentation:")
    # edge_detection_segmentation(image_path)
    edge_detection_segmentation(image_path)

    # Clustering-based Image Segmentation
    print("\nClustering-based Image Segmentation:")
    # clustering_based_segmentation(image_path, n_clusters=2)
    manual_clustering_based_segmentation(image_path, n_clusters=2)


if __name__ == "__main__":
    main()
