from ultralytics import YOLO

# 載入模型（可用 yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt / yolov8x.pt）
model = YOLO("yolo11n.pt")  # 可換成 yolov8n.pt 等

# 開始訓練
def main():
    model = YOLO("yolov8s.pt")
    model.train(data="C:/Users/user/Desktop/project/程式碼/test.yaml",
                epochs=300,
                imgsz=640,
                batch=2,
                lr0=0.001,
                weight_decay=0.001,
                mosaic=1.0,
                fliplr=0.5,
                translate=0.1,
                patience=100,
                pretrained=True,
                scale=0.5,
                project="runs/train",
                name="metal_yolo")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()