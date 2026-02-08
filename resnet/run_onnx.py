import onnx
import onnxruntime as ort
import numpy as np
import cv2

def proc_img(img_path,size=(90,180)):
    img=cv2.imread(img_path)
    img=cv2.resize(img,size)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=img.astype(np.float32)/255.0
    img=img.transpose(2,0,1)
    img=np.expand_dims(img,0)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,3,1,1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,3,1,1)

    img=(img-mean)/std

    return img.astype(np.float32)

onnx_model="/mnt/g//floder8/ccpd/resnet/ckp/province.onnx"
onnx.checker.check_model(onnx_model)

ort_session=ort.InferenceSession(onnx_model,providers=["CUDAExecutionProvider"])

img_path="/mnt/g/floder8/ccpd/dataset/resnet/province/val/25/0211494252874-91_81-251&454_526&548-542&559_265&545_245&454_522&468-25_0_1_14_29_30_33-165-70__0.jpg"
img=proc_img(img_path)
output=ort_session.run(None,{"input":img})[0]
pred=int(np.argmax(output,axis=1)[0])

print("class: ",pred)
print("logits: ",output[0])