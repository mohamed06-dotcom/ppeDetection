from ultralytics import YOLO

model_path = r"/runs\detect\train\weights\best.pt"
test_image = r"C:\Users\moham\PycharmProjects\ppeDetection\img.png"

# Load model
model = YOLO(model_path)

# Run inference
results = model(test_image, conf=0.25)  # adjust confidence threshold best 0.35 based on validation

results[0].show()

results[0].save(filename="result.jpg")