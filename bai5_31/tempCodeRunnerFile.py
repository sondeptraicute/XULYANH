import random

# Tạo bộ dữ liệu nha khoa giả lập
def generate_dental_data(num_samples=300):
    dental_data = []
    for _ in range(num_samples):
        # Tạo dữ liệu cho hai loại nha khoa: loại 0 và loại 1
        if random.random() < 0.5:
            dental_data.append([random.uniform(4.0, 8.0),  # feature 1
                                random.uniform(1.0, 5.0),  # feature 2
                                random.uniform(1.0, 7.0),  # feature 3
                                random.uniform(0.1, 3.0),  # feature 4
                                0])  # Nhãn cho loại 0
        else:
            dental_data.append([random.uniform(4.0, 8.0),
                                random.uniform(1.0, 5.0),
                                random.uniform(1.0, 7.0),
                                random.uniform(0.1, 3.0),
                                1])  # Nhãn cho loại 1
    return dental_data

# Tạo dữ liệu nha khoa
dental_data = generate_dental_data(300)  # Tạo 300 mẫu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data_dental = dental_data[:210]  # 70% cho huấn luyện
test_data_dental = dental_data[210:]    # 30% cho kiểm tra

# Hàm tính Gini Index
def gini_index(groups, classes):
    total_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[4] for row in group].count(class_val) / size
            score += proportion * proportion
        gini += (1.0 - score) * (size / total_instances)
    return gini

# Tạo cây CART (ví dụ đơn giản)
def cart(train):
    classes = list(set(row[4] for row in train))  # Lớp duy nhất
    # Chia nhóm dữ liệu (phân loại)
    # Thêm logic để chia nhóm và phát triển cây ở đây
    return "Cây CART (Gini Index) đã được tạo"

# Huấn luyện mô hình CART
cart_model = cart(train_data_dental)
print(cart_model)

# Kiểm tra độ chính xác trên tập kiểm tra
def predict_cart(test):
    # Thêm logic để dự đoán ở đây
    return [0] * len(test)  # Ví dụ dự đoán giả (tất cả về lớp 0)

predictions_cart = predict_cart(test_data_dental)
accuracy_cart = sum(1 for i in range(len(predictions_cart)) if predictions_cart[i] == test_data_dental[i][4]) / len(test_data_dental)
print(f'CART Accuracy on Dental Data: {accuracy_cart}')

# Tạo cây ID3 (Information Gain)
def id3(train):
    classes = list(set(row[-1] for row in train))  # Lớp duy nhất
    # Thêm logic để chia nhóm và phát triển cây ở đây
    return "Cây ID3 (Information Gain) đã được tạo"

# Huấn luyện mô hình ID3
id3_model = id3(train_data_dental)
print(id3_model)

# Kiểm tra độ chính xác trên tập kiểm tra
def predict_id3(test):
    # Thêm logic để dự đoán ở đây
    return [0] * len(test)  # Ví dụ dự đoán giả (tất cả về lớp 0)

predictions_id3 = predict_id3(test_data_dental)
accuracy_id3 = sum(1 for i in range(len(predictions_id3)) if predictions_id3[i] == test_data_dental[i][-1]) / len(test_data_dental)
print(f'ID3 Accuracy on Dental Data: {accuracy_id3}')
