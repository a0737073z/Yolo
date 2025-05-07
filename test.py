from ultralytics import YOLO
import cv2
import os

# 載入模型
model = YOLO(r"C:\Users\user\Desktop\project\code\yolo\runs\train\metal_yolo2\weights\best.pt")

# 原始圖片路徑
img_path = r"C:\Users\user\Desktop\0424\yolo\val\images\16.png"
img_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]  # "16"

# 執行預測
results = model.predict(source=img_path, save=True)

# 取得預測結果儲存的目錄
save_dir = results[0].save_dir

# 預測後的圖片是 jpg 格式
predicted_img_path = os.path.join(save_dir, img_name_no_ext + ".jpg")

# 讀取原始圖片
img = cv2.imread(img_path)

# 從結果中提取邊界框（boxes）
boxes = results[0].boxes  # 獲取邊界框信息

# 遍歷每個檢測到的物體，並根據邊界框裁剪圖片
for i, box in enumerate(boxes.xywh):
    # 獲取邊界框坐標 [x_center, y_center, width, height]
    x_center, y_center, w, h = box
    x_center, y_center, w, h = int(x_center), int(y_center), int(w), int(h)

    # 計算邊界框的左上角和右下角
    x1 = x_center - w // 2
    y1 = y_center - h // 2
    x2 = x_center + w // 2
    y2 = y_center + h // 2

    # 裁剪圖像，這會保留被框住的部分
    cropped_img = img[y1:y2, x1:x2]

    # 儲存裁剪後的圖片
    cropped_img_path = os.path.join(save_dir, f"cropped_{img_name_no_ext}_{i+1}.jpg")
    cv2.imwrite(cropped_img_path, cropped_img)
    print(f"保存裁剪圖像到: {cropped_img_path}")
