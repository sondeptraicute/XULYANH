import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from fcmeans import FCM

# Đọc ảnh vệ tinh
image_path = 'd:/XuLyAnh/bai7_9-11/image.png'  # Đường dẫn đến ảnh vệ tinh
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB

# Giảm kích thước ảnh
scale_percent = 50  # Tỉ lệ giảm kích thước
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Chuyển đổi ảnh thành dữ liệu 2D
pixel_values = image_resized.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Số lượng cụm
n_clusters = 3

# K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans_labels = kmeans.fit_predict(pixel_values)

# Fuzzy C-Means
fcm = FCM(n_clusters=n_clusters)
fcm.fit(pixel_values)
fcm_labels = fcm.predict(pixel_values)

# Agglomerative Hierarchical Clustering
ahc = AgglomerativeClustering(n_clusters=n_clusters)
ahc_labels = ahc.fit_predict(pixel_values)

# Tính toán số lượng điểm trong mỗi cụm
def print_cluster_sizes(labels, method_name):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Số lượng điểm trong các cụm ({method_name}):")
    for cluster, count in zip(unique, counts):
        print(f"Cụm {cluster}: {count} điểm")

# Hiển thị số so sánh trong terminal
print_cluster_sizes(kmeans_labels, "K-means")
print_cluster_sizes(fcm_labels, "Fuzzy C-Means")
print_cluster_sizes(ahc_labels, "Agglomerative Hierarchical Clustering")

# Hiển thị kết quả
plt.figure(figsize=(15, 5))

# Ảnh gốc
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Hiển thị nhãn của K-means
plt.subplot(1, 4, 2)
plt.imshow(kmeans_labels.reshape(image_resized.shape[:2]), cmap='tab20')  # Hiển thị nhãn K-means
plt.title('K-means Labels')
plt.axis('off')

# Hiển thị nhãn của Fuzzy C-Means
plt.subplot(1, 4, 3)
plt.imshow(fcm_labels.reshape(image_resized.shape[:2]), cmap='tab20')  # Hiển thị nhãn FCM
plt.title('Fuzzy C-Means Labels')
plt.axis('off')

# Hiển thị nhãn của AHC
plt.subplot(1, 4, 4)
plt.imshow(ahc_labels.reshape(image_resized.shape[:2]), cmap='tab20')  # Hiển thị nhãn AHC
plt.title('AHC Labels')
plt.axis('off')

plt.tight_layout()
plt.show()
