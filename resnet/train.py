import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

BATCH_SIZE=32
EPOCHS=50
LR=1e-3

def train_model(dataset_dir:str,num_calasses:int,model_save_path:str):
    """
    Docstring for train_model
    
    :param dataset: Description
    :type dataset: str
    :param num_calasses: Description
    :type num_calasses: int
    :param model_save_path: Description
    :type model_save_path: str
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ",device)
    train_transform=transforms.Compose([
        transforms.Resize((180,90)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])
    val_transform=transforms.Compose([
        transforms.Resize((180,90)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])

    train_dataset=datasets.ImageFolder(os.path.join(dataset_dir,"train"),transform=train_transform)
    val_dataset=datasets.ImageFolder(os.path.join(dataset_dir,"val"),transform=val_transform)

    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)

    print("classes: ",train_dataset.classes)

    model=models.resnet18(pretrained=True)
    in_features=model.fc.in_features
    model.fc=nn.Linear(in_features,num_calasses)
    model=model.to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=LR)
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

    def train_one_epoch(model,loader):
        model.train()
        running_loss=0.0
        correct=0

        for images,labels in tqdm(loader):
            images,labels=images.to(device),labels.to(device)

            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            _,preds=torch.max(outputs,1)
            correct+=torch.sum(preds==labels).item()
        
        epoch_loss=running_loss/len(loader)
        epoch_acc=correct/len(loader.dataset)

        return epoch_loss,epoch_acc
    
    def validate(model,loader):
        model.eval()
        running_loss=0.0
        correct=0

        with torch.no_grad():
            for images,labels in tqdm(loader):
                images,labels=images.to(device),labels.to(device)
                outputs=model(images)
                loss=criterion(outputs,labels)

                running_loss+=loss.item()
                _,preds=torch.max(outputs,1)
                correct+=torch.sum(preds==labels).item()
        
        epoch_loss=running_loss/len(loader)
        epoch_acc=correct/len(loader.dataset)

        return epoch_loss,epoch_acc
    
    best_acc=0.0
    best_model_wts=copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss,train_acc=train_one_epoch(model,train_loader)
        val_loss,val_acc=validate(model,val_loader)

        scheduler.step()

        print(f"train loss: {train_loss:.4f} Acc:{train_acc:.4f}")
        print(f"val loss: {val_loss:.4f} Acc:{val_acc:.4f}")

        if val_acc>best_acc:
            best_acc=val_acc
            best_model_wts=copy.deepcopy(model.state_dict())
            torch.save(best_model_wts,model_save_path)
            print("best model saved")
    print(f"\nBest val acc: {best_acc:.4f}")


if __name__=="__main__":
    # AZ01_DATASET="/mnt/g/floder8/ccpd/dataset/resnet/az01"
    # AZ01_NUM_CLS=34
    # AZ01_SAVE_PATH="/mnt/g//floder8/ccpd/resnet/ckp/az01_best.pth"
    PROVINCE_DATASET="/mnt/g/floder8/ccpd/dataset/resnet/province"
    PROVINCE_NUM_CLS=31
    PROVINCE_SAVE_PATH="/mnt/g//floder8/ccpd/resnet/ckp/province_best.pth"
    train_model(PROVINCE_DATASET,PROVINCE_NUM_CLS,PROVINCE_SAVE_PATH)


