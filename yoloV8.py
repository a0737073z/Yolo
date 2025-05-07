from ultralytics import YOLO


# 開始訓練
def main():
    model = YOLO("yolov8s.pt")
    model.train(data="C:/Users/user/Desktop/project/code/yolo/test.yaml",
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