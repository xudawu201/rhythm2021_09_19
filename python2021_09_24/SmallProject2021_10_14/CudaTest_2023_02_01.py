'''
Author: xudawu
Date: 2023-02-01 19:19:05
LastEditors: xudawu
LastEditTime: 2023-02-01 19:19:08
'''
import torch
# 1，查看gpu信息
if_cuda = torch.cuda.is_available()
print("torch.cuda.is_available()",if_cuda)

# GPU 的数量
gpu_count = torch.cuda.device_count()
print("torch.cuda.device_count()",gpu_count)

# 查看gpu名字，设备索引默认从0开始
print('torch.cuda.get_device_name(0)',torch.cuda.get_device_name(0))

# 返回当前设备索引；
print('torch.cuda.current_device()',torch.cuda.current_device())

# 2，将张量在gpu和cpu间移动
tensor = torch.rand((100,100))
tensor_gpu = tensor.to("cuda:0") # 或者 tensor_gpu = tensor.cuda()
print('tensor_gpu.device',tensor_gpu.device)
print('tensor_gpu.is_cuda',tensor_gpu.is_cuda)

tensor_cpu = tensor_gpu.to("cpu") # 或者 tensor_cpu = tensor_gpu.cpu() 
print('转为cpu',tensor_cpu.device)

# 3，清空cuda缓存
# 该方法在cuda超内存时十分有用
torch.cuda.empty_cache()