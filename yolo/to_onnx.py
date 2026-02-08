from ultralytics import YOLO

pt_model="/mnt/g/floder8/ccpd/yolo/runs/pose/train6/weights/best.pt"

# Load a model
model = YOLO(pt_model)  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom-trained model

# Export the model
model.export(format="onnx")