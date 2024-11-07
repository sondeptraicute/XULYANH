import numpy as np
import cv2
import os
import random

# Đường dẫn tới tệp ảnh
image_path = "d:/XuLyAnh/baitap6_7-11/image.png"

# Kiểm tra xem tệp ảnh có tồn tại hay không
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Tệp ảnh không tồn tại tại đường dẫn: {image_path}")

# Đọc ảnh đầu vào và chuyển sang không gian màu RGB
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Không thể đọc tệp ảnh tại đường dẫn: {image_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
data = image_rgb.reshape((-1, 3))  # Mỗi pixel là một điểm dữ liệu với 3 giá trị (R, G, B)

# Hàm tính khoảng cách Euclidean giữa hai điểm
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Chọn ngẫu nhiên một giá trị k trong phạm vi [2, 3, 4, 5]
k_values = [2, 3, 4, 5]
k = random.choice(k_values)
print(f"Chọn ngẫu nhiên k = {k}")

# Khởi tạo centroids ngẫu nhiên từ dữ liệu ảnh
centroids = data[np.random.choice(data.shape[0], k, replace=False)]

# Lặp thuật toán K-means
for _ in range(100):  # Giới hạn số lần lặp là 100
    # Gán các điểm vào các cụm gần nhất
    clusters = np.zeros(data.shape[0], dtype=int)
    for idx, point in enumerate(data):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        clusters[idx] = np.argmin(distances)

    # Cập nhật centroids mới dựa trên các điểm trong mỗi cụm
    new_centroids = np.zeros((k, 3))
    for j in range(k):
        cluster_points = data[clusters == j]
        if len(cluster_points) > 0:
            new_centroids[j] = np.mean(cluster_points, axis=0)
        else:
            # Xử lý cụm rỗng bằng cách chọn lại một điểm ngẫu nhiên làm centroid
            new_centroids[j] = data[np.random.choice(data.shape[0])]

    # Kiểm tra điều kiện dừng
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

# Tái tạo ảnh từ kết quả phân cụm
segmented_image = centroids[clusters].astype(np.uint8)
segmented_image = segmented_image.reshape(image_rgb.shape)

# Hiển thị ảnh phân cụm với cv2
cv2.imshow(f"Segmented Image - K = {k}", segmented_image)
cv2.waitKey(0)  # Đợi nhấn phím bất kỳ để đóng cửa sổ
cv2.destroyAllWindows()
