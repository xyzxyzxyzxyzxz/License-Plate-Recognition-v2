import cv2
import os
import numpy as np
import glob
import random
from tqdm import tqdm

import utils.alphabet as alphabet
import utils.coord_lp as coord_lp
import utils.lp_proc as lp_proc

IMG_FILE="01-90_88-214&487_400&552-397&550_213&552_204&485_388&483-0_0_33_30_26_21_30-144-12.jpg"
# IMG_FILE="02-90_85-173&466_452&541-452&553_176&556_178&463_454&460-0_0_6_26_15_26_32-68-53.jpg"
# IMG_FILE="01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24.jpg"
# CCPD_JPG_PATH="/mnt/d/floder8/chepai/CCPD2019/ccpd_base/"

# CCPD_JPG_PATH="/mnt/g/floder8/ccpd/img/test1"
CCPD_JPG_PATH="/mnt/g/floder8/ccpd/dataset/ccpd/ccpd_base"
RESNET_DATASET_PATH="/mnt/g/floder8/ccpd/dataset/resnet"

# def main():
#     img=cv2.imread(os.path.join("/mnt/g/floder8/ccpd/img",IMG_FILE))
#     x1,y1=list(map(int,IMG_FILE.split("-")[3].split("_")[2].split("&")))
#     x2,y2=list(map(int,IMG_FILE.split("-")[3].split("_")[1].split("&")))
#     x3,y3=list(map(int,IMG_FILE.split("-")[3].split("_")[0].split("&")))
#     x4,y4=list(map(int,IMG_FILE.split("-")[3].split("_")[3].split("&")))

#     pl_x1=min(x1,x2)
#     pl_y1=min(y1,y4)
#     pl_x3=max(x3,x4)
#     pl_y3=max(y2,y3)

#     pl_cut_x1=x1-pl_x1
#     pl_cut_y1=y1-pl_y1
#     pl_cut_x3=x3-pl_x1
#     pl_cut_y3=y3-pl_y1
#     pl_cut_x2=x2-pl_x1
#     pl_cut_y2=y2-pl_y1
#     pl_cut_x4=x4-pl_x1
#     pl_cut_y4=y4-pl_y1


#     print(pl_x3-pl_x1,pl_y3-pl_y1)
#     print(pl_cut_x1,pl_cut_y1)
#     print(pl_cut_x2,pl_cut_y2)
#     print(pl_cut_x3,pl_cut_y3)
#     print(pl_cut_x4,pl_cut_y4)

#     cv2.imshow("img",img)
#     pl=img[pl_y1:pl_y3+1,pl_x1:pl_x3+1]
#     pl[pl_cut_y1,pl_cut_x1]=(0,0,255)
#     pl[pl_cut_y2,pl_cut_x2]=(0,0,255)
#     pl[pl_cut_y3,pl_cut_x3]=(0,0,255)
#     pl[pl_cut_y4,pl_cut_x4]=(0,0,255)


#     cv2.imshow("pl",pl)

#     # 透视旋转
#     src_pts=np.float32(
#         [
#             [pl_cut_x1,pl_cut_y1],
#             [pl_cut_x4,pl_cut_y4],
#             [pl_cut_x3,pl_cut_y3],
#             [pl_cut_x2,pl_cut_y2]
#         ]
#     )

#     w,h=pl_x3-pl_x1,pl_y3-pl_y1

#     dst_pts=np.float32(
#         [
#             [0,0],
#             [w,0],
#             [w,h],
#             [0,h]
#         ]
#     )

#     to_M=cv2.getPerspectiveTransform(src_pts,dst_pts)
#     to_pl=cv2.warpPerspective(pl,to_M,(w,h))

#     # 车牌标准440x140
#     to_pl=cv2.resize(to_pl,(880,280))
    


#     gray_pl=cv2.cvtColor(pl,cv2.COLOR_BGR2GRAY)

#     to_pl=cv2.GaussianBlur(to_pl,ksize=(3,3),sigmaX=0)




#     # 直方图均衡化

#     to_pl=equalize_hist_color(to_pl)
    

#     # 提取蓝底白字
#     range_l=np.array([0,0,180],dtype=np.uint8)
#     range_u=np.array([179,110,255],dtype=np.uint8)
#     to_pl_hsv=cv2.cvtColor(to_pl,cv2.COLOR_BGR2HSV)
#     white_alpha=cv2.inRange(to_pl_hsv,range_l,range_u)

#     o_ker=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#     white_alpha_o=cv2.erode(white_alpha,o_ker,iterations=1)
#     white_alpha_o=cv2.dilate(white_alpha_o,o_ker,iterations=1)
#     diff_d=cv2.absdiff(white_alpha,white_alpha_o)


    

#     contours,hierarchy=cv2.findContours(white_alpha_o,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

#     cv2.drawContours(to_pl,contours,-1,(0,0,255),1)

#     # 通过contours提取每个字符 字符标准45x90
#     cnt_apl_rec=[]
#     for cnt_alp in contours:
#         alp_x,alp_y,alp_w,alp_h=cv2.boundingRect(cnt_alp)
#         if cv2.contourArea(cnt_alp)>8100 and cv2.contourArea(cnt_alp)<16200:
#             cnt_apl_rec.append([alp_x,alp_y,alp_w,alp_h])

#     cnt_apl_rec_sort=sorted(cnt_apl_rec,key=lambda b:b[0])

#     alphabet_list=alphabet.get_alphabet_num(IMG_FILE)
    
#     for i,rec in enumerate(cnt_apl_rec_sort):
#         print("w,h: ",rec[2],rec[3])
#         cv2.rectangle(to_pl,(rec[0],rec[1]),(rec[0]+rec[2],rec[1]+rec[3]),(0,0,255),1)
#         if i==0:
#             cv2.putText(to_pl,f"{i} {alphabet.get_province_str(alphabet_list[i])}",(rec[0],rec[1]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.2,color=(0,0,0),thickness=2,lineType=cv2.LINE_AA)
#         else:
#             cv2.putText(to_pl,f"{i} {alphabet.get_az01_str(alphabet_list[i])}",(rec[0],rec[1]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.2,color=(0,0,0),thickness=2,lineType=cv2.LINE_AA)

    

#     cv2.imshow("to pl",to_pl)
#     cv2.imshow("white alpha",white_alpha)
#     cv2.imshow("white alpha o",white_alpha_o)
#     cv2.imshow("gray pl",gray_pl)
#     cv2.imshow("diff o",diff_d)


#     cv2.waitKey(0)



def split_one_lp(lp:str):
    """
    Docstring for split_one_lp
    
    :param lp: Description
    :type lp: str
    """
    img=cv2.imread(os.path.join(CCPD_JPG_PATH,lp))
    coord14=coord_lp.get_lp_x1y1_to_x4y4(lp)
    coord_lp13=coord_lp.get_lp_x1y1_x3y3(coord14)

    coord_lp_cut14=coord_lp.get_lp_cut_x1y1_to_x4y4(coord14,coord_lp13)

    lp_x1,lp_y1,lp_x3,lp_y3=coord_lp13
    lp=img[lp_y1:lp_y3+1,lp_x1:lp_x3+1]

    lp_cut_x1,lp_cut_y1,lp_cut_x2,lp_cut_y2,lp_cut_x3,lp_cut_y3,lp_cut_x4,lp_cut_y4=coord_lp_cut14

        # 透视旋转
    src_pts=np.float32(
        [
            [lp_cut_x1,lp_cut_y1],
            [lp_cut_x4,lp_cut_y4],
            [lp_cut_x3,lp_cut_y3],
            [lp_cut_x2,lp_cut_y2]
        ]
    )

    w,h=lp_x3-lp_x1,lp_y3-lp_y1

    dst_pts=np.float32(
        [
            [0,0],
            [w,0],
            [w,h],
            [0,h]
        ]
    )

    to_M=cv2.getPerspectiveTransform(src_pts,dst_pts)
    to_lp=cv2.warpPerspective(lp,to_M,(w,h))

    # 车牌标准440x140
    to_lp=cv2.resize(to_lp,(880,280))
    to_lp=to_lp[50:-55,15:-35]
    return_lp=to_lp.copy()
    
    # gray_pl=cv2.cvtColor(pl,cv2.COLOR_BGR2GRAY)

    to_lp=cv2.GaussianBlur(to_lp,ksize=(5,5),sigmaX=0)

    # 直方图均衡化
    to_lp=equalize_hist_color(to_lp)
    # cv2.imshow("equalize hist",to_lp)

    # lp_gray=cv2.cvtColor(to_lp,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray",lp_gray)
    # _,white_alpha=cv2.threshold(lp_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 提取蓝底白字
    range_l=np.array([0,0,120],dtype=np.uint8)
    range_u=np.array([179,110,255],dtype=np.uint8)
    to_lp_hsv=cv2.cvtColor(to_lp,cv2.COLOR_BGR2HSV)
    white_alpha=cv2.inRange(to_lp_hsv,range_l,range_u)

    # cv2.imshow("white alpha",white_alpha)

    h_ker=cv2.getStructuringElement(cv2.MORPH_RECT,(1,180))
    h_ker1=cv2.getStructuringElement(cv2.MORPH_RECT,(1,15))
    w_ker=cv2.getStructuringElement(cv2.MORPH_RECT,(3,1))
    white_alpha_o=cv2.erode(white_alpha,h_ker1,iterations=1)
    white_alpha_o=cv2.dilate(white_alpha_o,h_ker,iterations=1)
    white_alpha_o=cv2.dilate(white_alpha_o,w_ker,iterations=1)
    # white_alpha_o=cv2.erode(white_alpha,h_ker,iterations=1)
    # white_alpha_o=cv2.erode(white_alpha,e_ker,iterations=1)
    # white_alpha_o=cv2.dilate(white_alpha_o,d_ker,iterations=1)
    # diff_d=cv2.absdiff(white_alpha,white_alpha_o)

    contours,hierarchy=cv2.findContours(white_alpha_o,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    d_c=cv2.cvtColor(white_alpha_o,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(d_c,contours,-1,(0,0,255),1)
    # cv2.imshow("draw c",d_c)


    # 通过contours提取每个字符 字符标准45x90
    cnt_apl_rec=[]
    for cnt_alp in contours:
        alp_x,alp_y,alp_w,alp_h=cv2.boundingRect(cnt_alp)
        if alp_w>15 and alp_h>150 and alp_w<150:
            cnt_apl_rec.append([alp_x,alp_y,alp_w,alp_h])

    return_recs=[]
    cnt_apl_rec_sort=sorted(cnt_apl_rec,key=lambda b:b[0])
    if len(cnt_apl_rec_sort)==8 and (cnt_apl_rec_sort[1][0]-(cnt_apl_rec_sort[0][0]+cnt_apl_rec_sort[0][2]))<20 and cnt_apl_rec_sort[0][2]<60:
        return_recs.append([cnt_apl_rec_sort[0][0],cnt_apl_rec_sort[0][1],
                            cnt_apl_rec_sort[1][0]+cnt_apl_rec_sort[1][2]-cnt_apl_rec_sort[0][0],
                            cnt_apl_rec_sort[1][1]+cnt_apl_rec_sort[1][3]-cnt_apl_rec_sort[0][1]])
        return_recs.extend(cnt_apl_rec_sort[2:])
    else:
        return_recs=cnt_apl_rec_sort

    if len(return_recs)==8:
        del return_recs[2]
    # cv2.waitKey(0)

    return return_lp,return_recs




def equalize_hist_color(img:cv2.typing.MatLike):
    """
    BGR to YCrCb, equalizeHist channel 0
    
    :param img: Description
    :type img: cv2.typing.MatLike
    """
    ycrcb_img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:,:,0]=cv2.equalizeHist(ycrcb_img[:,:,0])

    equalize_img=cv2.cvtColor(ycrcb_img,cv2.COLOR_YCrCb2BGR)

    return equalize_img


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

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ads = ['A', 'B', 'C', 'D', 'E',
       'F', 'G', 'H', 'J', 'K',
       'L', 'M', 'N', 'P', 'Q',
       'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z', '0',
       '1', '2', '3', '4', '5',
       '6', '7', '8', '9', 'O']
SAMPLE_NUM_TRAIN=200
SAMPLE_NUM_VAL=40

def split_alphabet_in_dataset(path:str):
    """
    Docstring for split_alphabet_in_dataset
    
    :param path: Description
    :type path: str
    """
    all_lp=get_all_lp(path)
    random.shuffle(all_lp)

    # 选择省份
    all_lp_s_by_p=lp_proc.select_lp_split_province(all_lp)
    # all_lp=all_lp_s_by_p[1]

    n=len(all_lp)
    n=int(n*0.8)
    resnet_train=all_lp[:n]
    resnet_val=all_lp[n:]
    resnet_train=resnet_train[:]
    resnet_val=resnet_val[:]

    dict_num_province={}
    dict_num_az01={}
    for i,_ in enumerate(provinces):
        dict_num_province[i]=0
        os.makedirs(os.path.join(RESNET_DATASET_PATH,"province","train",f"{i:02d}"),exist_ok=True)
        os.makedirs(os.path.join(RESNET_DATASET_PATH,"province","val",f"{i:02d}"),exist_ok=True)
    for i,_ in enumerate(ads):
        dict_num_az01[i]=0
        os.makedirs(os.path.join(RESNET_DATASET_PATH,"az01","train",f"{i:02d}"),exist_ok=True)
        os.makedirs(os.path.join(RESNET_DATASET_PATH,"az01","val",f"{i:02d}"),exist_ok=True)


    

    for lp in tqdm(resnet_train):
        lp_pers,lp_rec=split_one_lp(lp)
        lp_no_ext=lp.split(".")[0]
        
        if len(lp_rec)!=7:
            continue
        alphabet_list=alphabet.get_alphabet_num(lp)
        # show_lp(lp_pers,lp_rec,alphabet_list)
        for i ,rec in enumerate(lp_rec):
            alp_img=lp_pers[rec[1]:rec[1]+rec[3]+1,rec[0]:rec[0]+rec[2]+1]
            if i==0:
                file_name=dict_num_province[alphabet_list[i]]
                
                if file_name<SAMPLE_NUM_TRAIN:
                    cv2.imwrite(os.path.join(RESNET_DATASET_PATH,"province","train",f"{alphabet_list[i]:02d}",lp_no_ext+"__"+str(file_name)+".jpg"),alp_img)
                    dict_num_province[alphabet_list[i]]+=1
            else:
                file_name=dict_num_az01[alphabet_list[i]]
                
                if file_name<SAMPLE_NUM_TRAIN:
                    cv2.imwrite(os.path.join(RESNET_DATASET_PATH,"az01","train",f"{alphabet_list[i]:02d}",lp_no_ext+"__"+str(file_name)+".jpg"),alp_img)
                    dict_num_az01[alphabet_list[i]]+=1
        
    print("train========")
    print("province:")
    for i in dict_num_province:
        print(f"{i}-{provinces[i]}: {dict_num_province[i]}",end=" ")
    print(" ")
    print("az01:")
    for i in dict_num_az01:
        print(f"{i}-{ads[i]}: {dict_num_az01[i]}",end=" ")
    print(" ")
    
    for i,_ in enumerate(provinces):
        dict_num_province[i]=0
    for i,_ in enumerate(ads):
        dict_num_az01[i]=0


    for lp in tqdm(resnet_val):
        lp_pers,lp_rec=split_one_lp(lp)
        lp_no_ext=lp.split(".")[0]
        if len(lp_rec)!=7:
            continue
        alphabet_list=alphabet.get_alphabet_num(lp)
        # show_lp(lp_pers,lp_rec,alphabet_list)
        for i ,rec in enumerate(lp_rec):
            alp_img=lp_pers[rec[1]:rec[1]+rec[3]+1,rec[0]:rec[0]+rec[2]+1]
            if i==0:
                file_name=dict_num_province[alphabet_list[i]]
                
                if file_name<SAMPLE_NUM_VAL:
                    cv2.imwrite(os.path.join(RESNET_DATASET_PATH,"province","val",f"{alphabet_list[i]:02d}",lp_no_ext+"__"+str(file_name)+".jpg"),alp_img)
                    dict_num_province[alphabet_list[i]]+=1
            else:
                file_name=dict_num_az01[alphabet_list[i]]
                
                if file_name<SAMPLE_NUM_VAL:
                    cv2.imwrite(os.path.join(RESNET_DATASET_PATH,"az01","val",f"{alphabet_list[i]:02d}",lp_no_ext+"__"+str(file_name)+".jpg"),alp_img)
                    dict_num_az01[alphabet_list[i]]+=1

    print("val========")
    print("province:")
    for i in dict_num_province:
        print(f"{i}-{provinces[i]}: {dict_num_province[i]}",end=" ")
    print(" ")
    print("az01:")
    for i in dict_num_az01:
        print(f"{i}-{ads[i]}: {dict_num_az01[i]}",end=" ")
    print(" ")



def show_lp(lp_pers,lp_rec,alphabet_list):
    for i,rec in enumerate(lp_rec):
        # print("w,h: ",rec[2],rec[3])
        cv2.rectangle(lp_pers,(rec[0],rec[1]),(rec[0]+rec[2],rec[1]+rec[3]),(0,0,255),1)
        if i==0:
            cv2.putText(lp_pers,f"{i} {alphabet.get_province_str(alphabet_list[i])}",(rec[0],50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.2,color=(0,255,0),thickness=2,lineType=cv2.LINE_AA)
        else:
            cv2.putText(lp_pers,f"{i} {alphabet.get_az01_str(alphabet_list[i])}",(rec[0],50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.2,color=(0,255,0),thickness=2,lineType=cv2.LINE_AA)
    cv2.imshow("lp",lp_pers)
    cv2.waitKey(0)

if __name__=="__main__":
    # main()
    # lp_pers,lp_rec=split_one_lp(IMG_FILE)
    # for i,rec in enumerate(lp_rec):
    #     print("w,h: ",rec[2],rec[3])
    #     cv2.rectangle(lp_pers,(rec[0],rec[1]),(rec[0]+rec[2],rec[1]+rec[3]),(0,0,255),1)
    # cv2.imshow("lp",lp_pers)
    # cv2.waitKey(0)
    split_alphabet_in_dataset(CCPD_JPG_PATH)
    print("OK")