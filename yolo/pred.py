from ultralytics import YOLO

# Load a model
model = YOLO("/mnt/g/floder8/ccpd/yolo/runs/pose/train6/weights/best.onnx")  # pretrained YOLO26n model

# Run batched inference on a list of images
results = model(["/mnt/g/floder8/ccpd/img/01-90_88-214&487_400&552-397&550_213&552_204&485_388&483-0_0_33_30_26_21_30-144-12.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk
    print(boxes,keypoints)