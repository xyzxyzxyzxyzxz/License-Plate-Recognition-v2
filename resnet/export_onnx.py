import torch
import torch.nn as nn
from torchvision import models

def export_onnx(model_path:str,onnx_path:str,num_classes:int,is_az01:bool):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=models.resnet18(pretrained=False)
    model.fc=nn.Linear(model.fc.in_features,num_classes)

    state_dict=torch.load(model_path,map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    if is_az01:
        dummy_input=torch.rand(6,3,180,90).to(device)
    else:
        dummy_input=torch.rand(1,3,180,90).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        # dynamic_axes={
        #     "input":{0:"batch_size"},
        #     "output":{0:"batch_size"}
        # },
        export_params=True,
        external_data=False
    )

    print(f"ONNX model exported to {onnx_path}")


if __name__=="__main__":
    az01_model_path="/mnt/g//floder8/ccpd/resnet/ckp/az01_best.pth"
    az01_num_classes=34
    az01_onnx_path="/mnt/g//floder8/ccpd/resnet/ckp/az01.onnx"
    export_onnx(az01_model_path,az01_onnx_path,az01_num_classes,is_az01=True)
    province_model_path="/mnt/g//floder8/ccpd/resnet/ckp/province_best.pth"
    province_num_classes=31
    province_onnx_path="/mnt/g//floder8/ccpd/resnet/ckp/province.onnx"
    export_onnx(province_model_path,province_onnx_path,province_num_classes,is_az01=False)