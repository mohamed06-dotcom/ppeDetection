from ultralytics import YOLO

def main():
    model = YOLO("model/yolov8s.pt")

    model.train(
        data=r"C:\Users\moham\PycharmProjects\ppeDetection\data\ppe-pk.v3i.yolov8\data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,

        optimizer='AdamW',
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        close_mosaic=10,
        warmup_epochs=3.0,

        patience=30,
        workers=2,
        amp=True,
        plots=True
    )

if __name__=="__main__":
    main()