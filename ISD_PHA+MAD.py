'''
图片相似度检验
基于OpenCV的感知哈希算法结合MAD平均方误差算法用于相似度检测
From：Warmingeyes
Data：2020.05.22
Version：1.10
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

#计算方差
def getss(list):
    #计算平均值
    avg=sum(list)/len(list)
    #定义方差变量ss，初值为0
    ss=0
    #计算方差
    for l in list:
        ss+=(l-avg)*(l-avg)/len(list)
    #返回方差
    return ss

#获取每行像素平均值
def getdiff(img, Sidelength = 30):
    #定义边长
    Sidelength = 30

    #缩放图像
    img=cv2.resize(img,(Sidelength,Sidelength),interpolation=cv2.INTER_CUBIC)

    #灰度处理
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #RGB处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #avglist列表保存每行像素平均值
    avglist = []
    #计算每行均值，保存到avglist列表
    for i in range(Sidelength):
        avg=sum(gray[i])/len(gray[i])
        avglist.append(avg)
    #返回avglist平均值
    return avglist

#归一化处理
def Normalization(list):

    list_n = []
    k = sum(list)
    for i in list:
        list_n.append(i / k)
    #归一化

    return list_n

#均方差处理
def MSE(list):

    avglist=[]
    avg = sum(list)/len(list)
    for i in list:
        avglist.append(np.square(i - avg))
    mse = np.sqrt(sum(avglist) / len(list))
    #求得均方差

    return mse

#将list中的array转换为list方便提取
def list_arrat_to_list_list(list):
    for i in range(30):
        list[i] = list[[i][0]].tolist()

    return list

#读取测试图片
img1=cv2.imread("输入Figure1的绝对地址！")
diff_img1=getdiff(img1)
diff1=getss(diff_img1)
#print('img1:',diff1)

#读取测试图片
img2=cv2.imread("输入Figure2的绝对地址！")
diff_img2=getdiff(img2)
diff2 = getss(diff_img2)
#print('img2:',diff2)

#RGB方差归一化
diff1 = Normalization(diff1)
diff2 = Normalization(diff2)
#print('diff1_N:',diff1)
#print('diff1_N:',diff2)

#平方均误差的计算
mse1 = MSE(diff1)
mse2 = MSE(diff2)
print('PHA+MAD1:', mse1)
print('PHA+MAD2:', mse2)
print('PHA+MAD1-PHA+MAD2:', abs(mse1 - mse2))

#显示模块
print('''
The difference value of Figure1 and Figure2 smaller, more similar!
When the difference value smaller than 0.015, the figures are similar!
''')

if abs(mse1 - mse2) < 0.015:
    print("Similar!")
else:
    print("Different!")


#逐行方差绘制模块
#将全图逐行方差提取为RGB三通道的方差
diff_img1 = list_arrat_to_list_list(diff_img1)
diff_img2 = list_arrat_to_list_list(diff_img2)

diff_img1_R = []
diff_img1_G = []
diff_img1_B = []
diff_img2_R = []
diff_img2_G = []
diff_img2_B = []

for i in range(30):

    diff_img1_R.append(diff_img1[i][0])

    diff_img1_G.append(diff_img1[i][1])
    diff_img1_B.append(diff_img1[i][2])

    diff_img2_R.append(diff_img2[i][0])
    diff_img2_G.append(diff_img2[i][1])
    diff_img2_B.append(diff_img2[i][2])


#图形绘制
x=range(30)
#figure size设置
plt.figure("img_Red_Green_Blue",figsize=(15, 6))
plt.subplot(3,1,1)
plt.plot(x,diff_img1_R,marker="*",label="$Red01$")
plt.plot(x,diff_img2_R,marker="*",label="$Red02$")
plt.title("img_Red_Green_Blue")
#figure legend设置
plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)


x=range(30)
#plt.figure("img_Green")
plt.subplot(3,1,2)
plt.plot(x,diff_img1_G,marker="*",label="$Green01$")
plt.plot(x,diff_img2_G,marker="*",label="$Green02$")
#plt.title("img_Green")
plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)


x=range(30)
#plt.figure("img_Blue")
plt.subplot(3,1,3)
plt.plot(x,diff_img1_G,marker="*",label="$Blue01$")
plt.plot(x,diff_img2_G,marker="*",label="$Blue02$")
#plt.title("img_Blue")
plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
