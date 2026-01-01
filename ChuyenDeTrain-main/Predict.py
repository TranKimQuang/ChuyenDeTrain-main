from ultralytics import YOLO
import cv2

# 1. Load model mới nhất (đảm bảo file này nằm cùng thư mục)
model = YOLO("best.pt")

# 2. Predict với ngưỡng tin cậy thấp hơn để bắt được nhiều vật thể hơn
# Thêm tham số conf=0.25 để cải thiện việc "lúc được lúc không"
results = model.predict("img_test/test111.png", conf=0.25)

# 3. Hiển thị
for r in results:
    # Lưu kết quả ra file để xem cho rõ nếu cửa sổ hiện lên quá nhanh
    r.save(filename='result_test.jpg')
    # Hiện cửa sổ
    im_array = r.plot()  # Trả về mảng ảnh đã vẽ box
    cv2.imshow("Kiem tra Model", im_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()