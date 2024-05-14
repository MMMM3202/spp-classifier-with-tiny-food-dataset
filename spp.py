from math import floor, ceil
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidPooling2d(nn.Module):

    def __init__(self, num_level, pool_type='max_pool'):
        super(SpatialPyramidPooling2d, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type

    def forward(self, x):

        N, C, H, W = x.size()

        for i in range(self.num_level):
            level = i + 1
            kernel_size = (ceil(H / level), ceil(W / level))
            stride = (ceil(H / level), ceil(W / level))
            padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))

            if self.pool_type == 'max_pool':
                tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            else:
                tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)

            if i == 0:
                res = tensor

            else:
                res = torch.cat((res, tensor), 1)

        return res


class SPPNet(nn.Module):
    def __init__(self, num_level=2, pool_type='max_pool'):
        super(SPPNet, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type
        self.feature = nn.Sequential(nn.Conv2d(3, 64, 5),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(64, 32, 3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(32, 32, 3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(32, 32, 3),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2)
                                     )
        # num_grid = 1 + 4 + 9 = 14
        self.num_grid = self._cal_num_grids(num_level)
        self.spp_layer = SpatialPyramidPooling2d(num_level)
        self.linear = nn.Sequential(nn.Linear(self.num_grid * 32, 512),
                                    nn.Linear(512, 5))
        self.softmax = nn.Softmax(dim=1)


    def _cal_num_grids(self, level):
        count = 0
        for i in range(level):
            count += (i + 1) * (i + 1)
        return count

    def forward(self, x):
        x = self.feature(x)
        x = self.spp_layer(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    a = torch.rand((1, 3, 64, 64))
    net = SPPNet()
    output = net(a)
    print(output)

# from math import floor, ceil
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class SpatialPyramidPooling2d(nn.Module):
#
#     def __init__(self, num_level, pool_type='max_pool'):
#         super(SpatialPyramidPooling2d, self).__init__()
#         self.num_level = num_level
#         self.pool_type = pool_type
#
#     def forward(self, x):
#
#         N, C, H, W = x.size()
#
#         print('多尺度提取信息，并进行特征融合...')
#         print()
#         for i in range(self.num_level):
#             level = i + 1
#             print('第', level, '次计算池化核：')
#             kernel_size = (ceil(H / level), ceil(W / level))
#             print('kernel_size: ', kernel_size)
#             stride = (ceil(H / level), ceil(W / level))
#             print('stride: ', stride)
#             padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))
#             print('padding: ', padding)
#             print()
#
#             print('进行最大池化并将提取特征展开：')
#             if self.pool_type == 'max_pool':
#                 tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
#             else:
#                 tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
#
#             if i == 0:
#                 res = tensor
#                 print('展开大小为： ', res.size())
#                 print()
#             else:
#                 res = torch.cat((res, tensor), 1)
#                 print('合并为： ', res.size())
#                 print()
#         return res
#
#
# class SPPNet(nn.Module):
#     def __init__(self, num_level=3, pool_type='max_pool'):
#         super(SPPNet, self).__init__()
#         self.num_level = num_level
#         self.pool_type = pool_type
#         self.feature = nn.Sequential(nn.Conv2d(3, 64, 3),
#                                      nn.ReLU(),
#                                      nn.MaxPool2d(2),
#                                      nn.Conv2d(64, 64, 3),
#                                      nn.ReLU())
#         # num_grid = 1 + 4 + 9 = 14
#         self.num_grid = self._cal_num_grids(num_level)
#         self.spp_layer = SpatialPyramidPooling2d(num_level)
#         self.linear = nn.Sequential(nn.Linear(self.num_grid * 64, 512),
#                                     nn.Linear(512, 5))
#
#     def _cal_num_grids(self, level):
#         count = 0
#         for i in range(level):
#             count += (i + 1) * (i + 1)
#         return count
#
#     def forward(self, x):
#         print('x初始大小为：')
#         N, C, H, W = x.size()
#         print('N:', N, ' C:', C, ' H', H, ' W:', W)
#         print()
#
#         x = self.feature(x)
#         print('x经过卷积、激活、最大池化、卷积、激活变成：')
#         N, C, H, W = x.size()
#         print('64(conv)->62(maxpool)->31(conv)->29')
#         print('N:', N, ' C:', C, ' H', H, ' W:', W)
#         print()
#
#         print('x进行空间金字塔池化：')
#         x = self.spp_layer(x)
#
#         print('空间金字塔池化后，x进入全连接层：')
#         x = self.linear(x)
#         return x
#
#
# if __name__ == '__main__':
#     a = torch.rand((1, 3, 64, 64))
#     net = SPPNet()
#     output = net(a)
#     print(output)
