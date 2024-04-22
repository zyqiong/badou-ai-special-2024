'''
@author linrenkang
透视变换
'''

import cv2
import numpy as np


class MyImg:
    # 原图像
    img = None
    # 变换图像
    chImg = None
    dstW = 0
    dstH = 0

    # 计算矩阵
    A = None
    B = None
    T = None

    # 输入的个数
    POINT_CNT = 4
    A_ROW = 8
    A_COL = 8
    B_ROW = 8
    B_COL = 1

    def __init__(self, path):
        self.img = cv2.imread(path)
        self.A = np.zeros((self.A_ROW, self.A_COL))
        self.B = np.zeros((self.B_ROW, self.B_COL))

    # ------------ 私有函数 -------------
    # 根据输入的坐标，计算 A 和 B 矩阵
    def _SetAB(self, src, dst):
        assert src.shape[0] >= self.POINT_CNT and dst.shape[0] >= self.POINT_CNT
        for i in range(self.POINT_CNT):
            # 原坐标与目的坐标
            srcX, srcY = src[i, 0], src[i, 1]
            dstX, dstY = dst[i, 0], dst[i, 1]
            # 设置参数
            self.A[i * 2] = [srcX, srcY, 1, 0, 0, 0, -srcX * dstX, -srcY * dstX]
            self.A[i * 2 + 1] = [0, 0, 0, srcX, srcY, 1, -srcX * dstY, -srcY * dstY]
            self.B[i * 2] = dstX
            self.B[i + 2 + 1] = dstY

        self.A = np.mat(self.A)
        self.B = np.mat(self.B)

    # 根据 A 和 B 矩阵，计算 T
    def _SetT(self):
        tmp = self.A.I * self.B
        self.T = np.array(self.A.I * self.B).T[0]
        self.T = np.insert(self.T, self.T.shape[0], values=1, axis=0)
        self.T = self.T.reshape((3, 3))

        # 正确 T 矩阵
        # self.T = np.array([[-5.01338334e-01, -1.35357643e+00,5.82386716e+02],
        #  [2.27373675e-16,-4.84035391e+00,1.38781980e+03],
        #  [-0.00000000e+00,-4.14856327e-03,1.00000000e+00]])


    # 设置目标图像大小
    def _SetSize(self, h, w):
        self.dstH = h
        self.dstW = w
        self.chImg = np.zeros((h, w, self.img.shape[2]))

    # 获取新的坐标
    def _GetNewPoint(self, x, y):
        inputM = np.mat([x, y, 1])[0].T
        inputM = np.mat(inputM)
        print("inputM: ", inputM)
        print("T: ", self.T)
        outputM = self.T * inputM
        outputM = np.array(outputM).T[0]
        xNew = outputM[0]
        yNew = outputM[1]
        zNew = outputM[2]
        print("outputM: ", outputM)
        print("outputM0: ", int(xNew / zNew))
        print("outputM1: ", int(yNew / zNew))
        # assert 0
        return int(xNew / zNew), int(yNew / zNew)

    def _ExpectHandle(self, x, y):
        x = x if x > 0 else 0
        y = y if y > 0 else 0

        x = x if x < self.dstW else self.dstW - 1
        y = y if y < self.dstH else self.dstH - 1

        return x, y

    # 输出图片
    def _ShowPic(self, img):
        print("Picture: ", img)
        print("Shape: ", img.shape)
        cv2.imshow("Picture", img)
        cv2.waitKey(0)

    # 输出图片
    def _ShowPicName(self, img, name):
        print("Picture: " + name, img)
        print("Shape: ", img.shape)
        cv2.imshow("Picture: " + name, img)
        cv2.waitKey(0)

    # 输出直方图与合并图片
    def _ShowMergePic(self, imgSrc, chImg):
        self._ShowPic(chImg)
        cv2.imshow("Picture", np.hstack([imgSrc, chImg]))
        cv2.waitKey(0)

    #----------------- 导出接口 -------------
    # 训练透视变换函数
    def PerTransformDetail(self, src, dst):
        src = np.float32(src)
        dst = np.float32(dst)
        self._SetAB(src, dst)
        self._SetT()
        # print("PerTransformDetail T: ", self.T)
        # PerTransformDetail T: [[2.36411251e+00  9.98180836e-01 - 6.40096595e+02]
        #     [-4.95340812e-01  1.14593770e+00 - 7.05010445e+01]
        #     [2.85547407e-03 3.34475690e-04 1.00000000e+00]]
        self._Check(src, dst)

    def _Check(self, src, dst):
        print("check src: ", src)
        dstTmp = np.zeros(src.shape)
        for i in range(src.shape[0]):
            x, y = src[i, 0], src[i, 1]
            xNew, yNew = self._GetNewPoint(x, y)
            dstTmp[i, 0], dstTmp[i, 1] = xNew, yNew
        print("Check dstTmp: ", dstTmp)
        print("Check dst: ", dst)

    # 根据训练的 T 进行透视变换
    def WarpPerspectiveDetail(self, dstH, dstW):
        h, w = self.img.shape[:2]
        self._SetSize(dstH, dstW)
        for y in range(h):
            for x in range(w):
                xNew, yNew = self._GetNewPoint(x, y)
                xNew, yNew = self._ExpectHandle(xNew, yNew)
                self.chImg[yNew, xNew] = self.img[y, x]

    # 训练透视变换函数
    def PerTransform(self, src, dst):
        src = np.float32(src)
        dst = np.float32(dst)
        self.T = cv2.getPerspectiveTransform(src, dst)
        print("PerTransform T: ", self.T)

    def WarpPerspective(self, dstH, dstW):
        self.chImg = cv2.warpPerspective(self.img, self.T, (dstH, dstW))

    # 输出原图
    def ShowSrcImg(self):
        self._ShowPic(self.img)

    # 输出转换后的图像（彩色图）
    def ShowDstImg(self):
        # self._ShowMergePic(self.img, self.chImg)
        self._ShowPic(self.chImg)

'''
以下为透视变换的应用
'''

if __name__ == "__main__":
    myImg = MyImg('pic/photo1.jpg')
    # 输出原图
    # myImg.ShowSrcImg()

    # 设置源目地址
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]

    # src = [[207, 151], [517, 285], [17, 601], [343, 731]]
    # dst = [[0, 0], [337, 0], [0, 488], [337, 488]]
    myImg.PerTransformDetail(src, dst)
    # myImg.WarpPerspectiveDetail(500, 500)

    # myImg.PerTransform(src, dst)
    # myImg.WarpPerspective(500, 500)
    # myImg.ShowDstImg()