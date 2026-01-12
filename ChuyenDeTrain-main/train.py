from ultralytics import YOLO
import os

# 1️⃣ Thiết lập đường dẫn dataset để kiểm tra dữ liệu
# Lưu ý: Hãy đảm bảo đường dẫn này chính xác với thư mục mới của bạn
dataset_path = "C:/Users/DELL/Downloads/ChuyenDeTrain-main/ChuyenDeTrain-main/dataset/"

# Kiểm tra số lượng nhãn đã gán để chắc chắn dataset đã sẵn sàng
if os.path.exists(os.path.join(dataset_path, "labels/train")):
    labels_train = [f for f in os.listdir(os.path.join(dataset_path, "labels/train")) if f.lower().endswith(".txt")]
    total_labels = 0
    for label_file in labels_train:
        with open(os.path.join(dataset_path, "labels/train", label_file)) as f:
            total_labels += len(f.readlines())
    print(f"--- THÔNG TIN DATASET ---")
    print(f"Số lượng file nhãn trong tập train: {len(labels_train)}")
    print(f"Tổng số đối tượng (objects) đã gán nhãn: {total_labels}")
    print(f"--------------------------")
else:
    print("Cảnh báo: Không tìm thấy thư mục labels/train. Hãy kiểm tra lại đường dẫn dataset_path!")

# 2️⃣ Load YOLOv8 pre-trained model (Khởi tạo từ đầu cho dataset mới)
model = YOLO("yolov8n.pt")

# 3️⃣ Bắt đầu quá trình Train
results = model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    name="yolov8_final",
    device='cpu',
    augment=True,
    exist_ok=True
)

print("--- HOÀN THÀNH ---")
print("Model tốt nhất đã được lưu tại: runs/detect/yolov8_final/weights/best.pt")
print("Hãy copy file best.pt đó vào thư mục chứa App của bạn.")