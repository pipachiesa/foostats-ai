from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('models/best.pt')  # Load a pre-trained model (you can specify the path to your model)

# Run inference on a source
results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
print("-------------------------")
for box in results[0].boxes:
    print(box)