from utils import alphabet

def select_lp_split_province(all_lp:list):
    """
    split lp by different provindce
    
    :param all_lp: Description
    :type all_lp: list
    """
    
    # print(len(all_lp))

    all_lp_split_province=[]
    for _ in alphabet.provinces:
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
        str_e_p+=f"{alphabet.provinces[i]}: {len(e_p)},"

    print(str_e_p)

    return all_lp_split_province