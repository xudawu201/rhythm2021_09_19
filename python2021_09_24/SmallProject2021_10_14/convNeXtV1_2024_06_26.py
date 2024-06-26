'''
Author: xudawu
Date: 2024-06-26 17:28:19
LastEditors: xudawu
LastEditTime: 2024-06-26 17:29:22
'''
"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
 
 
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
 
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
 
 
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    官方实现的LN是默认对最后一个维度进行的，这里是对channel维度进行的，所以单另设一个类。
    """
 
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
 
 
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # layer scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
 
        x = shortcut + self.drop_path(x)
        return x
 
 
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 3,padding_size: int=0, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # stem为最初的下采样部分
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4,padding=padding_size),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
 
        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2,padding=int(padding_size/2)))
            self.downsample_layers.append(downsample_layer)
 
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
 
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
 
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)
 
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
 
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x
 
 
def convnext_tiny(in_chans,padding_size,num_classes: int):
    """
    初始化一个小型的ConvNeXt模型。

    ConvNeXt是一种基于Transformer架构的卷积神经网络模型，本函数初始化的是其小型变体。
    
    参数:
    num_classes: int - 模型输出的类别数量。这决定了模型最后一层的输出维度。
    
    返回:
    一个初始化完成的ConvNeXt小型模型实例。
    """
    # 根据预定义的配置初始化ConvNeXt模型
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    # 有4个stage，depths代表了每个stage中block的数量
    # dims代表了下采样四个卷积层的输出通道数,同时也代表每个block的输入通道数
    model = ConvNeXt(in_chans=in_chans,padding_size=padding_size,depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model
 
 
def convnext_small(in_chans,padding_size,num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(in_chans=in_chans,padding_size=padding_size,depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model
 
 
def convnext_base(in_chans,padding_size,num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(in_chans=in_chans,padding_size=padding_size,depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model
 
 
def convnext_large(in_chans,padding_size,num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(in_chans=in_chans,padding_size=padding_size,depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes)
    return model
 
 
def convnext_xlarge(in_chans,padding_size,num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(in_chans=in_chans,padding_size=padding_size,depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes)
    return model

# 输入通道数
in_chans=1
# padding大小
padding_size=0
# 输出类别
output_size=4
# 实例化convnext_tiny
convnext_tiny_model=convnext_tiny(in_chans,padding_size,output_size)
# print(convnext_tiny_model)
# 实例化convnext_base
convnext_base_model=convnext_base(in_chans,padding_size,output_size)

# 测试用例,输入1个3通道224x224的图片
input_size=1
input=torch.randn(input_size,in_chans,224,224)
output=convnext_tiny_model(input)
print(output.shape)
print(output)

print('- '*10,'分割线 ','- '*10)

# softmax
output_softmax=torch.nn.functional.softmax(output,dim=1)
print(output_softmax.shape)
print(output_softmax)

# 定义优化器
convnext_tiny_optimizer=torch.optim.AdamW(convnext_tiny_model.parameters(),lr=0.001)
convnext_base_optimizer=torch.optim.AdamW(convnext_base_model.parameters(),lr=0.001)
# 定义损失函数
criterion=torch.nn.MSELoss()

# 使用cpu训练
# device = torch.device('cpu')
# 使用GPU训练,当电脑中有多张显卡时，使用第一张显卡
device = torch.device('cuda:0')

# 模型转到GPU
# convnext_tiny_model=convnext_tiny_model.to(device)
convnext_base_model=convnext_base_model.to(device)

# 定义输出目标
target=torch.randn(input_size,output_size)
# 转到device
target = target.to(device)

import time

# 测试网络输出性能，基于时间评价
def testTime():
    
    for epoch in range(100):
        # 计时
        start_time=time.time()
        # 输入数据转到GPU
        input = input.to(device)

        # 获得网络输出
        # output=convnext_tiny_model(input)
        output=convnext_base_model(input)
        # 结束计时
        end_time = time.time()
        print('epoch:',epoch,'output:',output,'网络输出耗时：', end_time - start_time)

# 测试网络输出性能，基于时间评价
testTime()

print('- '*10,'分割线 ','- '*10)

# 测试网络训练性能，基于时间评价
def testTrainTime():
    for epoch in range(100):

        # 计时
        start_time=time.time()
        # 训练数据转到GPU
        input = input.to(device)
        # 网络输出
        # output=convnext_tiny_model(input)
        output=convnext_base_model(input)
        # 计算loss
        loss=criterion(output,target)
        
        # 梯度清零、反向传播、更新参数
        # convnext_tiny_optimizer.zero_grad()
        convnext_base_optimizer.zero_grad()
        loss.backward()
        # convnext_tiny_optimizer.step()
        convnext_base_optimizer.step()

        end_time = time.time()
        print('epoch:',epoch,'loss:',loss.item(),'训练耗时：', end_time - start_time)

# 测试网络训练性能，基于时间评价
testTrainTime()

# 存储模型
# 模型转到cpu
# convnext_base_model.to('cpu')
# # 模型路径
# modelFilePath_str = 'test.model'
# # 存储模型
# torch.save(convnext_base_model,modelFilePath_str)
# # 加载模型
# testModel=torch.load(modelFilePath_str)

print('- '*10,'分割线 ','- '*10)

# 实例化convnext_base
# convnext_base_model=convnext_base(4)
# print(convnext_base_model)
