import os
import glob
import random
import cv2
import shutil

import preprocess.utils.coord_lp as coord_lp

CCPD_JPG_PATH="/mnt/d/floder8/chepai/CCPD2019/ccpd_base/"

# CCPD_JPG_PATH="/mnt/d/floder8/chepai/CCPD2019/test/"
YOLO_DATASET_DIR="/mnt/g/floder8/ccpd/dataset/yolo"

SAMPLE_NUM=1000


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E',
             'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q',
             'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E',
       'F', 'G', 'H', 'J', 'K',
       'L', 'M', 'N', 'P', 'Q',
       'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z', '0',
       '1', '2', '3', '4', '5',
       '6', '7', '8', '9', 'O']


def get_all_lp(path:str)->list[str]:
    """
    Docstring for get_all_lp
    
    :param path: The ccpd jpg floder path
    :type path: str
    :return: All jpg file in path
    :rtype: list[str]
    """
    all_lp=glob.glob(os.path.join(path,"*.jpg"))
    all_lp=[lp.split("/")[-1] for lp in all_lp]
    return all_lp



def select_lp_split_province():
    all_lp=get_all_lp(CCPD_JPG_PATH)
    # print(len(all_lp))

    all_lp_split_province=[]
    for _ in provinces:
        all_lp_split_province.append([])

    print(all_lp_split_province,len(all_lp_split_province))

    for lp in all_lp:
        province_alpha=lp.split("-")[-3].split("_")[0]
        province_num=int(province_alpha)
        try:
            all_lp_split_province[province_num].append(lp)
        except IndexError:
            print(province_alpha,"NUMBER ERROR")


    str_e_p=""
    for i,e_p in enumerate(all_lp_split_province):
        str_e_p+=f"{provinces[i]}: {len(e_p)},"

    print(str_e_p)

    return all_lp_split_province

def get_lp_yolo(sample_num:int):
    all_lp=get_all_lp(CCPD_JPG_PATH)
    random.shuffle(all_lp)
    yolo_lp=all_lp[:sample_num]
    return yolo_lp

def prepare_for_yolo(dataset_dir:str):
    yolo_lp=get_lp_yolo(SAMPLE_NUM)
    lp_num=len(yolo_lp)
    print(f"{lp_num} license plate")

    train_lp_num=int(0.8*lp_num)
    val_lp_num=lp_num-train_lp_num

    train_yolo_lp=yolo_lp[:train_lp_num]
    val_yolo_lp=yolo_lp[train_lp_num:]


    os.makedirs(os.path.join(dataset_dir,"labels","train"),exist_ok=True)
    os.makedirs(os.path.join(dataset_dir,"labels","val"),exist_ok=True)
    os.makedirs(os.path.join(dataset_dir,"images","train"),exist_ok=True)
    os.makedirs(os.path.join(dataset_dir,"images","val"),exist_ok=True)

    for lp in train_yolo_lp:
        label=proc_every_lp_label(lp)
        lp_txt=lp.split(".")[0]+".txt"

        with open(os.path.join(dataset_dir,"labels","train",lp_txt),"w") as f:
            f.write(label+"\n")
        shutil.copy(os.path.join(CCPD_JPG_PATH,lp),os.path.join(dataset_dir,"images","train",lp))


    for lp in val_yolo_lp:
        label=proc_every_lp_label(lp)
        lp_txt=lp.split(".")[0]+".txt"
        
        with open(os.path.join(dataset_dir,"labels","val",lp_txt),"w") as f:
            f.write(label+"\n")
        shutil.copy(os.path.join(CCPD_JPG_PATH,lp),os.path.join(dataset_dir,"images","val",lp))
    
    print("OK")

    # 01-90_85-274&361_472&420-475&416_277&422_271&357_469&351-0_0_25_29_33_26_30-165-31        

def proc_every_lp_label(lp:str):
    coord=coord_lp.get_lp_x1y1_to_x4y4(lp)
    x1,y1,x2,y2,x3,y3,x4,y4=coord
    lp_x1,lp_y1,lp_x3,lp_y3=coord_lp.get_lp_x1y1_x3y3(coord)
    w_lp,h_lp=coord_lp.get_lp_wh(coord)
    
    img=cv2.imread(os.path.join(CCPD_JPG_PATH,lp))
    h,w=img.shape[:2]

    lp_x1_01,lp_x3_01=lp_x1/w,lp_x3/w
    lp_y1_01,lp_y3_01=lp_y1/h,lp_y3/h

    lp_xc_01=(lp_x1_01+lp_x3_01)/2
    lp_yc_01=(lp_y1_01+lp_y3_01)/2
    w_lp_01=w_lp/w
    h_lp_01=h_lp/h
    x1_01,x2_01,x3_01,x4_01=x1/w,x2/w,x3/w,x4/w
    y1_01,y2_01,y3_01,y4_01=y1/h,y2/h,y3/h,y4/h

    # cls xc yc w h kpt0_x kpt0_y v0 kpt1_x kpt1_y v1 ...
    label=f"0 {lp_xc_01} {lp_yc_01} {w_lp_01} {h_lp_01} {x1_01} {y1_01} {x2_01} {y2_01} {x3_01} {y3_01} {x4_01} {y4_01}"

    return label


def main():
    # select_lp_split_province()
    prepare_for_yolo(YOLO_DATASET_DIR)


if __name__=="__main__":
    main()
    
