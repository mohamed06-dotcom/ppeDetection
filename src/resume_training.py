from ultralytics import YOLO

def main():
    checkpoint = r"C:\Users\moham\PycharmProjects\ppeDetection\runs\detect\train\weights\last.pt"
    model = YOLO(checkpoint)

    model.train(resume=True)


if __name__=="__main__":
    main()