
import pandas as pd                                   
from sklearn.datasets import load_iris                
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier       
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
import matplotlib.pyplot as plt                       
from sklearn import tree                              


iris = load_iris()  # Tải bộ dữ liệu Iris
X = iris.data  # Lấy dữ liệu đầu vào (các đặc trưng) từ bộ dữ liệu
y = iris.target  # Lấy nhãn (loại hoa) từ bộ dữ liệu

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Chia dữ liệu thành 80% cho tập huấn luyện và 20% cho tập kiểm tra, với random_state để đảm bảo tính tái lập

cart_model = DecisionTreeClassifier(criterion='gini', random_state=42)  # Khởi tạo mô hình cây quyết định với tiêu chí Gini
cart_model.fit(X_train, y_train)  # Huấn luyện mô hình trên tập huấn luyện

# Đánh giá mô hình CART
y_pred_cart = cart_model.predict(X_test)  # Dự đoán nhãn cho tập kiểm tra
cart_accuracy = accuracy_score(y_test, y_pred_cart)  # Tính độ chính xác của mô hình

print(f'Độ chính xác của CART (Gini index): {cart_accuracy:.2f}')  # In ra độ chính xác của mô hình CART
print("Báo cáo phân loại CART:")  # In ra tiêu đề cho báo cáo phân loại
print(classification_report(y_test, y_pred_cart))  # In ra báo cáo phân loại cho mô hình CART
print("Ma trận Confusion CART:")  # In ra tiêu đề cho ma trận nhầm lẫn
print(confusion_matrix(y_test, y_pred_cart))  # In ra ma trận nhầm lẫn cho mô hình CART

id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42)  # Khởi tạo mô hình cây quyết định với tiêu chí entropy (ID3)
id3_model.fit(X_train, y_train)  # Huấn luyện mô hình trên tập huấn luyện

# Đánh giá mô hình ID3
y_pred_id3 = id3_model.predict(X_test)  # Dự đoán nhãn cho tập kiểm tra
id3_accuracy = accuracy_score(y_test, y_pred_id3)  # Tính độ chính xác của mô hình
print(f'Độ chính xác của ID3 (Information Gain): {id3_accuracy:.2f}')  # In ra độ chính xác của mô hình ID3
print("Báo cáo phân loại của ID3:")  # In ra tiêu đề cho báo cáo phân loại
print(classification_report(y_test, y_pred_id3))  # In ra báo cáo phân loại cho mô hình ID3
print("Ma trận Confusion của ID3:")  # In ra tiêu đề cho ma trận nhầm lẫn
print(confusion_matrix(y_test, y_pred_id3))  # In ra ma trận nhầm lẫn cho mô hình ID3

plt.figure(figsize=(10, 8))  # Tạo một hình vẽ với kích thước 10x8 inches
tree.plot_tree(cart_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
# Vẽ cây quyết định cho mô hình CART, với các đặc trưng và nhãn lớp
plt.title("Bộ phân lớp CART (Gini Index)")  # Đặt tiêu đề cho biểu đồ
plt.show()  # Hiển thị biểu đồ

plt.figure(figsize=(10, 8))  # Tạo một hình vẽ với kích thước 10x8 inches
tree.plot_tree(id3_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
# Vẽ cây quyết định cho mô hình ID3, với các đặc trưng và nhãn lớp
plt.title("Bộ phân lớp ID3 (Information Gain)")  # Đặt tiêu đề cho biểu đồ
plt.show()  # Hiển thị biểu đồ