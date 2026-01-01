import customtkinter as ctk
from ultralytics import YOLO
from tkinter import filedialog, messagebox
import cv2
import os
from PIL import Image, ImageTk

# Cấu hình giao diện
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SmartCheckoutApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI SMART CHECKOUT - GIỎ HÀNG THÔNG MINH")
        self.geometry("1100x700")
        self.resizable(False, False)  # Cố định kích thước cửa sổ

        self.model = YOLO("best.pt")
        self.product_prices = self.load_prices()

        # --- GRID LAYOUT ---
        self.grid_columnconfigure(0, weight=3)  # Cột ảnh (Rộng)
        self.grid_columnconfigure(1, weight=1)  # Cột Panel (Hẹp hơn)
        self.grid_rowconfigure(0, weight=1)

        # --- PANEL TRÁI: HIỂN THỊ ẢNH ---
        self.image_frame = ctk.CTkFrame(self, corner_radius=0)
        self.image_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.image_label = ctk.CTkLabel(self.image_frame, text="CHƯA CÓ DỮ LIỆU\nVui lòng bấm 'Quét sản phẩm'",
                                        font=("Arial", 16))
        self.image_label.pack(expand=True, fill="both")

        # --- PANEL PHẢI: THÔNG TIN HÓA ĐƠN ---
        self.side_panel = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color="#2c3e50")
        self.side_panel.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")

        ctk.CTkLabel(self.side_panel, text="DANH SÁCH MẶT HÀNG", font=("Arial", 20, "bold")).pack(pady=20)

        # Danh sách sản phẩm (Dạng bảng cuộn)
        self.items_list = ctk.CTkTextbox(self.side_panel, width=280, height=350, font=("Consolas", 13))
        self.items_list.pack(pady=10, padx=10)
        self.items_list.configure(state="disabled")

        # Khu vực Tổng tiền
        self.total_frame = ctk.CTkFrame(self.side_panel, fg_color="#1a252f", corner_radius=10)
        self.total_frame.pack(pady=20, padx=15, fill="x")

        ctk.CTkLabel(self.total_frame, text="TỔNG CỘNG:", font=("Arial", 14)).pack(pady=(10, 0))
        self.total_val_label = ctk.CTkLabel(self.total_frame, text="0 VND", font=("Arial", 28, "bold"),
                                            text_color="#2ecc71")
        self.total_val_label.pack(pady=(0, 10))

        # Nút bấm thao tác
        self.btn_scan = ctk.CTkButton(self.side_panel, text="QUÉT SẢN PHẨM", command=self.select_image,
                                      height=45, font=("Arial", 14, "bold"), fg_color="#27ae60", hover_color="#219150")
        self.btn_scan.pack(pady=10, padx=20, fill="x")

        self.btn_clear = ctk.CTkButton(self.side_panel, text="XÓA HÓA ĐƠN", command=self.clear_data,
                                       height=40, fg_color="#e74c3c", hover_color="#c0392b")
        self.btn_clear.pack(pady=5, padx=20, fill="x")

    def load_prices(self):
        # Tự động lấy giá từ model
        prices = {}
        for id, name in self.model.names.items():
            try:
                price = int(name.split('-')[-1].replace('VND', '').strip())
                prices[name] = price
            except:
                prices[name] = 0
        return prices

    def clear_data(self):
        self.image_label.configure(image=None, text="CHƯA CÓ DỮ LIỆU")
        self.items_list.configure(state="normal")
        self.items_list.delete("1.0", "end")
        self.items_list.configure(state="disabled")
        self.total_val_label.configure(text="0 VND")

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return

        results = self.model.predict(file_path, conf=0.25)[0]
        img = results.orig_img.copy()

        detected_items = []
        total_bill = 0

        # Xử lý nhận diện
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label_name = self.model.names[cls_id]
            price = self.product_prices.get(label_name, 0)

            total_bill += price
            short_name = label_name.split('-')[0]
            detected_items.append(f"{short_name[:15]:<15} | {price:>8,}đ")

            # Vẽ khung lên ảnh
            cv2.rectangle(img, (x1, y1), (x2, y2), (46, 204, 113), 3)
            cv2.putText(img, short_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113), 2)

        # 1. Cập nhật Panel danh sách (Bên phải)
        self.items_list.configure(state="normal")
        self.items_list.delete("1.0", "end")
        self.items_list.insert("end", f"{'SẢN PHẨM':<15} | {'GIÁ':>9}\n")
        self.items_list.insert("end", "-" * 30 + "\n")
        for item in detected_items:
            self.items_list.insert("end", item + "\n")
        self.items_list.configure(state="disabled")

        # 2. Cập nhật Tổng tiền
        self.total_val_label.configure(text=f"{total_bill:,} VND")

        # 3. Hiển thị ảnh vào Frame bên trái
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Resize ảnh để vừa khít Panel trái nhưng giữ tỉ lệ
        display_w, display_h = 750, 650
        img_pil.thumbnail((display_w, display_h))

        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=img_pil.size)
        self.image_label.configure(image=img_tk, text="")


if __name__ == "__main__":
    app = SmartCheckoutApp()
    app.mainloop()