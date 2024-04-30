import serial
import numpy as np
import cv2
import matplotlib.pyplot as plt

from uwb_positioning_packs.find_angel_distance import get_angle_distance,draw_angle_distance
from uwb_positioning_packs.find_tag_position import gradient_descent
from uwb_positioning_packs.custom_kalmanfilter import custom_kalman1D

def main():
    #定義錨點位置
    #將00的點偏移到中心
    offset=[360,360]
    center=[
        [0,20,0],
        [-10,-10,0],
        [10,-10,0]
    ]
    center_color=[
        [255,0,0],
        [0,255,0],
        [0,0,255]
    ]

    device_tag=serial.Serial("COM6",115200)
    # device_tag=None
    #draw a whole dark image
    image=np.zeros((720,720,3),np.uint8)
    #draw the centers
    for i in range(len(center)):
        cv2.circle(image,np.sum((center[i][0:2],offset),axis=0),5,center_color[i],0)
    #畫角度參考線
    cv2.arrowedLine(image, np.sum(((0,0),offset),0), np.sum(((0,100),offset),0), (255,255,255), 1)
    
    #畫格子，沒使用
    for i in range(0,0,1):
        cv2.line(image,(50*i,0),(50*i,720),255,1)
        cv2.line(image,(0,50*i),(720,50*i),255,1)

    cv2.imshow("a",image)
    if cv2.waitKey(0) & 0xff ==ord('q'):
        return

    #分別建立各錨點的卡爾曼綠波
    k_r0=custom_kalman1D()
    k_r1=custom_kalman1D()
    k_r2=custom_kalman1D()

    print("strat")
    #main procedure
    
    while True:
        #copy an image from the original image
        current_img=image.copy()
        #put tag_pos into temp list
        X=np.array([1,1,1])
        #initial circle radus
        # ensure every radius cound be found 
        device_tag.reset_input_buffer()
        r0=-1
        r1=-1
        r2=-1
        while r0==-1 or r1==-1 or r2==-1:
            #從COM取資料
            result=device_tag.readline()
            #解碼資料
            result=result.decode()
            #確保資料完整
            if len(result)!=13:
                continue
            try:
                 #got anchor index and each radius and storing it
                #取錨點index
                anchor_index=int(result[0])
                #取與其的距離
                r_temp=float(result[3:])
                #公尺變公分
                r_temp=r_temp*100
                if r_temp<0.0 and r_temp>700:
                    continue
                if anchor_index==0:
                    r0=r_temp
                elif anchor_index==1:
                    r1=r_temp
                elif anchor_index==2:
                    r2=r_temp
            except:
                continue

        #卡爾曼濾波器找平滑
        r0=k_r0.renew_and_getdata(r0)
        r1=k_r1.renew_and_getdata(r1)
        r2=k_r2.renew_and_getdata(r2)
        #轉成int 因為opencv畫圖只能用int
        r=np.array([r0,r1,r2]).astype(int)
        print("r="+str(r))
        #find local minimum x,y to fit the equtions
        try:
            #找近似值X
            X=gradient_descent(X,center,r)
            # X=sympy_solve_equtions(X,center,r)
        except:
            continue
        # if X[2]>220:
        #     continue
        print("x="+str(X))
    
        # print(tag_position)
        #原本預計找多個近似值X再求平均，但太耗時
        tag_position=np.asarray(X)
        tag_position=tag_position.astype(int).reshape((3,))
        
        #找夾角與距離
        angel_distance=get_angle_distance(tag_position)


        #using kalman filter
        # tag_position=Kalman1D(tag_position)

        #畫標籤的位置，省略Z軸
        cv2.circle(current_img,np.sum((tag_position[0:2],offset),0),5,(0,255,0),3)
        #畫標籤到各錨點的線與圓形，並寫上距離
        draw_angle_distance(current_img,np.sum((tag_position[0:2],offset),0),angel_distance)
        for i , center_pos in enumerate(center):
            cv2.circle(current_img,np.sum((offset,center_pos[0:2]),0),r[i],center_color[i],1)
            # cv2.line(current_img,center_pos[0:2],tag_position[0:2],255,1)            
            # cv2.putText(current_img, str("{:.1f}".format(np.sqrt(np.sum((center_pos-tag_position)**2)))) ,np.array((center_pos[0:2]+tag_position[0:2])/2).astype(int),cv2.FONT_HERSHEY_SIMPLEX,0.8,255)
        cv2.imshow("a",current_img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        #plot
        # plt.plot(tag_position[0],tag_position[1],'go')
        # plt.show()

main()