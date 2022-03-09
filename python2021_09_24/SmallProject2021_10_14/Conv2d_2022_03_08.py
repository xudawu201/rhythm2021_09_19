'''
Author: xudawu
Date: 2022-03-08 09:23:54
LastEditors: xudawu
LastEditTime: 2022-03-08 15:29:08
'''
'''
1.
conv2d_shape:
input(N,C_in,H_in,W_in) # batch, channel , height , width
output(N,C_out,H_out,W_out)
C_out=out_channels
H_out/W_out = (H_out/W_out - kennel_size + 2*padding) / stride + 1
2.
MaxPool2d_shape:
同conv2d
3.
Linear_shape
input(N,C_in,H_in,W_in)
Linear(in_features,out_features)
in_features=C_in*H_in*W_in
'''
import torch
x = torch.randn(2, 3, 4, 5)  # batch, channel , height , width
#(2,3,4,5)总共2个/行，这2个/行里每个有3个/行,这3行有4行5列
# print(x)
print(x.shape)
in_channels = 3
out_channels = 3
kernel_size = 3
stride = 1
padding=int((kernel_size - stride) / 2)  #使卷积后图片尺寸不变化,即height和width不变化
m = torch.nn.Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
print(m)
y = m(x)
print(y.shape)
# print(y)
m = torch.nn.MaxPool2d(3)
print(m)
y = m(y)
# print(y)
print('y_shape:',y.shape)
# 将四维张量转换为二维张量之后，才能作为全连接层的输入
y = y.view(x.size(0), -1)
print(y.shape)
print(y)
m=torch.nn.Linear(in_features=1*1*3,out_features=2)
print(m)
y = m(y)
print(y.shape)
print(y)