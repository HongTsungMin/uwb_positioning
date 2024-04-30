import numpy as np
import cv2
def get_angle_distance(target_point):
    #預設機器中心點為(0.0)
    if len(target_point)==3: #不接受三維空間
        target_point=np.array(target_point[0:2])#預設輸入格式為[x,y,z]，只取[x,y]
    elif len(target_point)>=4: 
        return "no support dim is 4"
    print(target_point)
    reference_vector=np.array([0,1])#設定參考向量軸
    innerproduct=np.inner(reference_vector,target_point)#計算內積
    # print(innerproduct)
    distance=np.linalg.norm(target_point)#計算與中心點的距離
    # print(distance)
    angel=np.arccos((innerproduct/distance))#計算與參考軸的夾角
    angel=angel/np.pi*180*(-1 if target_point[0]<0 else 1)#弧度轉角度
    print(f"angel:{int(angel*1000)/1000}',distance:{int(distance*1000)/1000}")#捨位並輸出
    return [angel,distance]

#get_angle_distance([10,9,30])

def draw_angle_distance(current_image,tag_position,angel_distance):
    #畫線，從中心到目標
    color=255
    thickness=3
    offset=(360,360)
    cv2.arrowedLine(current_image, np.sum(((0,0),offset),0), tag_position, color, thickness)
    
    #文字
    text = str(angel_distance[0])+"'  "+str(angel_distance[1])
    org = (100,50)#寫在中心點上方
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255,255,255)
    thickness = 1
    lineType = cv2.LINE_AA#反鋸齒  LINE_8無反鋸齒
    cv2.putText(current_image, text, org, fontFace, fontScale, color, thickness, lineType)
