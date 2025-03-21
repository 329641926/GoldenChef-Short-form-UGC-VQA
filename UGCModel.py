import clip
import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
import numpy as np
import cv2


class Bottleneck(nn.Module):
    # ResBlock
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        # 如果是空间分支(slow分支)的第一层卷积
        if head_conv == 1:
            # 没有时间卷积核大小
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:  # 如果是时间分支(fast分支)的第一层卷积
            # 有时间卷积核大小为3
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        # 两个分支的第二层卷积相同
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                               padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        # 两个分支的第三层卷积相同
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # 是否降采样
        self.downsample = downsample
        # 步长
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class SlowFast(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3]):
        super(SlowFast, self).__init__()
        # 时间分支
        self.fast_inplanes = 8
        # fast分支第一个卷积，对空间降采样
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        # 池化空间降采样
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_layer = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)

        # 堆叠res2、res3、res4、res5,每一次堆叠只降采样一次
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3)
        self.fast_res3 = self._make_layer_fast(block, 16, layers[1], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(block, 32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(block, 64, layers[3], stride=2, head_conv=3)
        self.conv3d = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1, 7, 7), stride=(1, 1, 1),
                                padding=(0, 0, 0))

    def forward(self, input):
        # 将采集的图像以时间步长为2的方式送入fast分支
        x = self.FastPath(input)

        return x

    def FastPath(self, input):

        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)

        res2 = self.fast_res2(pool1)

        res3 = self.fast_res3(res2)  # [N, 64, frames, 28, 28]

        res4 = self.fast_res4(res3)

        res5 = self.fast_res5(res4)

        x = self.conv3d(res5)
        x_size = x.shape
        x = x.reshape(x_size[0], x_size[1], -1)
        x = x.permute(0, 2, 1)

        return x

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        # 如果需要空间降采样，对进入fast分支网络层的特征进行直接降采样
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.fast_inplanes, planes * block.expansion, kernel_size=1,
                          stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        # 添加ResBlock，带降采样
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        # 堆叠ResBlock，不带降采样
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)


def base_quality_regression(in_channels, middle_channels, out_channels):
    regression_block = nn.Sequential(
        nn.Linear(in_channels, middle_channels),
        nn.ReLU(),
        nn.Linear(middle_channels, out_channels),

    )
    return regression_block


def base_quality_regression(in_channels, middle_channels, out_channels):
    regression_block = nn.Sequential(
        nn.Linear(in_channels, middle_channels),
        nn.ReLU(),
        nn.Linear(middle_channels, out_channels),

    )
    return regression_block


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 自适应计算卷积核大小
        k_size = int(abs((np.log2(channels) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1  # 保证为奇数

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        y = self.avg_pool(x)  # [B,C,1,1]

        # Excitation
        y = y.squeeze(-1).transpose(1, 2)  # [B,1,C]
        y = self.conv(y)  # [B,1,C]
        y = self.sigmoid(y)  # [B,1,C]

        y = y.transpose(1, 2).unsqueeze(-1)  # 恢复维度 [B,C,1,1]
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.eca = ECA(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.eca(x)  # 插入注意力
        x = self.conv2(x)
        return x + identity


class CrossAttentionPro(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # 交叉投影层
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # 动态卷积
        self.dynamic_conv = nn.Conv2d(
            num_heads, num_heads,
            kernel_size=3, padding=1, groups=num_heads)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        B, N, C = x.shape

        # 生成Query和Key-Value对
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, H, L, D)

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 动态卷积增强
        attn = self.dynamic_conv(attn)

        # 特征聚合
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(combined)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)  # 通道注意力
        x = x * self.sa(x)  # 空间注意力
        return x


class ViT_32_Swin_Tiny_Fast_densenet121_Model(torch.nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ViT_32_Swin_Tiny_Fast_densenet121_Model, self).__init__()

        ViT_B_16, _ = clip.load("ViT-B/32")
        clip_vit_b_pretrained_features = ViT_B_16.visual
        self.text_feature_extraction = clip_vit_b_pretrained_features
        model = models.swin_v2_t(weights='Swin_V2_T_Weights.DEFAULT')
        self.spatial_feature_extraction = torch.nn.Sequential(*list(model.children())[:-1])
        densenet121 = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        self.densenet = torch.nn.Sequential(*list(densenet121.children())[:-1])
        self.slowfast = SlowFast()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cross_attn = CrossAttentionPro(dim=512, num_heads=8)
        self.dense_regression = nn.Linear(in_features=768, out_features=1024)
        self.cross_attn_dense = CrossAttentionPro(dim=1024, num_heads=8)
        self.SEBlock = SEBlock(1024)
        self.base_quality_regression = base_quality_regression(in_channels=3072, middle_channels=128, out_channels=1)
        self.device = device

    def forward(self, x, y):
        # input dimension: batch x frames x 3 x height x width

        x_size = x.shape
        z = y
        B, T, C, H, W = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]
        y = y.reshape(-1, C, H, W)
        x = x.reshape(-1, C, H, W)
        text_feature = self.text_feature_extraction(y).view(B, T, -1)  # torch.Size([batch_size,frames, 512])
        motion = self.slowfast(z.permute(0, 2, 1, 3, 4))

        cross_attn_feature = self.cross_attn(motion, text_feature)  # torch.Size([batch_size,frames, 512])
        motion_feature = torch.mean(motion + cross_attn_feature, dim=1)  # torch.Size([batch_size, 512])

        cross_attn_feature = self.cross_attn(text_feature, motion)  # torch.Size([batch_size,frames, 512])
        text_feature = torch.mean(text_feature + cross_attn_feature, dim=1)  # torch.Size([batch_size, 512])

        y_features = self.spatial_feature_extraction(x)  # torch.Size([batch_size*frames, 768, 1, 1])
        spatial_feature = self.dense_regression(y_features.reshape(B, T, -1))

        dense = self.densenet(y)  # torch.Size([8, 1024, 7, 7])
        dense = self.SEBlock(dense)  # torch.Size([8, 1024, 7, 7])
        dense = self.avg_pool(dense)  # torch.Size([8, 1024])
        compression_feature = dense.view(B, T, -1)

        spatial_attn = self.cross_attn_dense(spatial_feature, compression_feature)
        spatial_feature_attn = torch.mean(spatial_feature + spatial_attn, dim=1)
        dense_attn = self.cross_attn_dense(compression_feature, spatial_feature)
        compression_feature_attn = torch.mean(compression_feature + dense_attn, dim=1)

        final_feature = torch.cat([text_feature, spatial_feature_attn, compression_feature_attn, motion_feature], dim=1)
        score = self.base_quality_regression(final_feature)

        return score.squeeze(1)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT_32_Swin_Tiny_Fast_densenet121_Model().to(device)
    model = model.float()
    input_sample = (torch.randn(1, 12, 3, 224, 224).to(device),  # x
                    torch.randn(1, 12, 3, 224, 224).to(device))  # z

    flops, params = profile(model, inputs=input_sample)
    # flops= count_gflops_with_fvcore(model, input_sample)
    print(f"Total GFLOPs: {flops / 1e9}")
