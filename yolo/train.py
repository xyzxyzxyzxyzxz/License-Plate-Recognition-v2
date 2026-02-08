from ultralytics import YOLO

# Load a model
# model = YOLO("yolo26n.yaml")  # build a new model from YAML
model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/mnt/g/floder8/ccpd/dataset/yolo/dataset.yaml",
                       epochs=250, imgsz=900,save_period=10,batch=24,cos_lr=True
                       )