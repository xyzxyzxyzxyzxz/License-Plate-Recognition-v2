

def get_lp_x1y1_to_x4y4(lp:str):
    """
    x1y1:top left,x2y2:bottum left,x3y3:bottum right,x4y4:top right
    
    :param lp: license plate
    :type lp: str
    """
    x1,y1=list(map(int,lp.split("-")[3].split("_")[2].split("&")))
    x2,y2=list(map(int,lp.split("-")[3].split("_")[1].split("&")))
    x3,y3=list(map(int,lp.split("-")[3].split("_")[0].split("&")))
    x4,y4=list(map(int,lp.split("-")[3].split("_")[3].split("&")))
    return (x1,y1,x2,y2,x3,y3,x4,y4)

def get_lp_x1y1_x3y3(coord:list[int]):
    """
    license plate cutting top left, bottum right
    
    :param coord: [x1,y1,x2,y2,x3,y3,x4,y4]
    :type coord: list[int]
    """
    x1,y1,x2,y2,x3,y3,x4,y4=coord

    lp_x1=min(x1,x2)
    lp_y1=min(y1,y4)
    lp_x3=max(x3,x4)
    lp_y3=max(y2,y3)

    return (lp_x1,lp_y1,lp_x3,lp_y3)

def get_lp_cut_x1y1_to_x4y4(coord:list[int],coord_lp:list[int]):
    """
    x1y1 to x4y4 in cutting lisence plate
    
    :param coord: [x1,y1,x2,y2,x3,y3,x4,y4]
    :type coord: list[int]

    :param coord_lp: [pl_x1,pl_y1,pl_x3,pl_y3]
    :type coord_lp: list[int]
    """
    x1,y1,x2,y2,x3,y3,x4,y4=coord
    pl_x1,pl_y1,pl_x3,pl_y3=coord_lp
    pl_cut_x1=x1-pl_x1
    pl_cut_y1=y1-pl_y1
    pl_cut_x3=x3-pl_x1
    pl_cut_y3=y3-pl_y1
    pl_cut_x2=x2-pl_x1
    pl_cut_y2=y2-pl_y1
    pl_cut_x4=x4-pl_x1
    pl_cut_y4=y4-pl_y1

    return (pl_cut_x1,pl_cut_y1,
            pl_cut_x2,pl_cut_y2,
            pl_cut_x3,pl_cut_y3,
            pl_cut_x4,pl_cut_y4
            )

def get_lp_wh(coord:list[int]):
    """
    license plate w h
    
    :param coord: [x1,y1,x2,y2,x3,y3,x4,y4]
    :type coord: list[int]
    """
    lp_x1,lp_y1,lp_x3,lp_y3=get_lp_x1y1_x3y3(coord)

    w=lp_x3-lp_x1
    h=lp_y3-lp_y1

    return (w,h)

