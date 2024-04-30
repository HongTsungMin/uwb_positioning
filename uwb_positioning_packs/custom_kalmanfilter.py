import numpy as np

class custom_kalman1D:
    def __init__(self):
        n = 10
        self.Q = 1e-5  # 過程噪聲協方差
        self.R = 0.01**2  # 觀測噪聲協方差
        self.xhat = np.zeros(n)  # 估計值
        self.xhat = [0,0]
        self.P = np.zeros(n)  # 估計值的協方差
        self.P =[0]
        self.xhatminus = np.zeros(n)  # 預測值
        self.xhatminus =[0]
        self.Pminus = np.zeros(n)  # 預測值的協方差
        self.Pminus =[0]
        self.K = np.zeros(n)  # 卡爾曼增益
        self.K =[0]
        self.xhat[0] = 0.0
        self.P[0] = 1.0

    def renew_and_getdata(self,raw_data):
        # 預測
        # xhatminus[k] = xhat[k-1]
        self.xhatminus[0]=self.xhat[0]
        # Pminus[k] = P[k-1] + Q
        self.Pminus[0]=self.P[0]+self.Q
        # 更新
        # K[k] = Pminus[k] / (Pminus[k] + R)
        self.K[0]=self.Pminus[0]/(self.Pminus[0]+self.R)
        # xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        self.xhat[1]=self.xhatminus[0]+self.K[0]*(raw_data-self.xhatminus[0])
        # P[k] = (1 - K[k]) * Pminus[k]
        self.P[0] = (1 - self.K[0]) * self.Pminus[0]
        self.xhat[0]=self.xhat[1]
        return self.xhat[0]