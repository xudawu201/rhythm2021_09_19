'''
Author: xudawu
Date: 2022-05-20 15:12:07
LastEditors: xudawu
LastEditTime: 2022-05-20 15:26:36
'''
import torch
# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# in_features:size of each input sample
# out_features:size of each output sample
a=[[1,2],[3,4],[5,6]]
print(a)
a=torch.FloatTensor(a)
in_features=2 #输入的特征数,等于矩阵的列数
out_features=3 #输出的特征数
m = torch.nn.Linear(in_features, out_features)
print(a[0].size())
print(a[0])
output = m(a[0])
print(output)