import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel(blurred_image):
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    _, sobel_magnitude = cv2.threshold(sobel_magnitude, 100, 255, cv2.THRESH_BINARY)
    return sobel_magnitude

def apply_prewitt(blurred_image):
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_x = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_y)
    prewitt_magnitude = cv2.magnitude(prewitt_x, prewitt_y)
    _, prewitt_magnitude = cv2.threshold(prewitt_magnitude, 100, 255, cv2.THRESH_BINARY)
    return prewitt_magnitude

def apply_roberts(blurred_image):
    kernel_roberts_x = np.array([[1, 0], [0, -1]])
    kernel_roberts_y = np.array([[0, 1], [-1, 0]])
    roberts_x = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_roberts_x)
    roberts_y = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_roberts_y)
    roberts_magnitude = cv2.magnitude(roberts_x, roberts_y)
    _, roberts_magnitude = cv2.threshold(roberts_magnitude, 50, 255, cv2.THRESH_BINARY)
    return roberts_magnitude

def apply_canny(blurred_image):
    canny_edges = cv2.Canny(blurred_image, 100, 200)
    return canny_edges

# Đọc ảnh vệ tinh màu
image = cv2.imread('d:/XuLyAnh/baitap_12-11/image.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB để hiển thị với Matplotlib
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang thang độ xám

# Bước 1: Làm mờ ảnh với bộ lọc Gaussian
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Bước 2: Áp dụng các toán tử phát hiện biên
sobel_magnitude = apply_sobel(blurred_image)
prewitt_magnitude = apply_prewitt(blurred_image)
roberts_magnitude = apply_roberts(blurred_image)
canny_edges = apply_canny(blurred_image)

# Bước 3: Hiển thị kết quả
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image (Color)")

plt.subplot(2, 3, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title("Gaussian Blurred Image")

plt.subplot(2, 3, 3)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title("Sobel Edge Detection")

plt.subplot(2, 3, 4)
plt.imshow(prewitt_magnitude, cmap='gray')
plt.title("Prewitt Edge Detection")

plt.subplot(2, 3, 5)
plt.imshow(roberts_magnitude, cmap='gray')
plt.title("Roberts Edge Detection")

plt.subplot(2, 3, 6)
plt.imshow(canny_edges, cmap='gray')
plt.title("Canny Edge Detection")

plt.tight_layout()
plt.show()