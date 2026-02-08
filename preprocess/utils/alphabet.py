# 01-90_88-214&487_400&552-397&550_213&552_204&485_388&483-0_0_33_30_26_21_30-144-12.jpg
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ads = ['A', 'B', 'C', 'D', 'E',
       'F', 'G', 'H', 'J', 'K',
       'L', 'M', 'N', 'P', 'Q',
       'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z', '0',
       '1', '2', '3', '4', '5',
       '6', '7', '8', '9', 'O']

def get_alphabet_num(lp:str):
    """
    Docstring for get_alphabet_num
    
    :param lp: lisence plate
    :type lp: str
    """
    alphabet_list=lp.split("-")[-3].split("_")
    assert len(alphabet_list)==7
    alphabet_list=list(map(int,alphabet_list))
    return alphabet_list

def get_province_str(idx:int):
    """
    Docstring for get_province_str
    
    :param idx: Description
    :type idx: int
    """
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    return provinces[idx]

def get_az01_str(idx:int):
    """
    Docstring for get_az01_str
    
    :param idx: Description
    :type idx: int
    """
    ads = ['A', 'B', 'C', 'D', 'E',
       'F', 'G', 'H', 'J', 'K',
       'L', 'M', 'N', 'P', 'Q',
       'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z', '0',
       '1', '2', '3', '4', '5',
       '6', '7', '8', '9', 'O']
    return ads[idx]