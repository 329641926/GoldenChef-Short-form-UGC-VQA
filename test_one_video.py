import time
import argparse

from thop import profile
from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
from PIL import Image
import cv2
import clip
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


def base_quality_regression(in_channels, middle_channels, out_channels):
    regression_block = nn.Sequential(
        nn.Linear(in_channels, middle_channels),
        nn.ReLU(),
        nn.Linear(middle_channels, out_channels),
    )
    return regression_block


def calculate_frame_difference(frame1, frame2):
    # 计算帧差并转换为灰度图
    return cv2.cvtColor(cv2.absdiff(frame1, frame2), cv2.COLOR_BGR2GRAY)


def sample_patches_from_frame_difference(frame_diff, original_frame, grid_size=7, block_size=32):
    grid_height = frame_diff.shape[0] // grid_size
    grid_width = frame_diff.shape[1] // grid_size
    selected_positions = []

    for i in range(grid_size):
        for j in range(grid_size):
            start_y = i * grid_height
            end_y = (i + 1) * grid_height
            start_x = j * grid_width
            end_x = (j + 1) * grid_width

            grid_region = frame_diff[start_y:end_y, start_x:end_x]
            max_s = 0
            max_position = (0, 0)

            for m in range(0, grid_height, block_size):
                for n in range(0, grid_width, block_size):
                    small_block = grid_region[m:m + block_size, n:n + block_size]
                    s = np.sum(small_block)

                    if s > max_s:
                        max_s = s
                        max_position = (start_y + m, start_x + n)

            selected_positions.append(max_position)

    sampled_patches = []
    for y, x in selected_positions:
        if y + block_size > original_frame.shape[0]:
            y = original_frame.shape[0] - block_size
        if x + block_size > original_frame.shape[1]:
            x = original_frame.shape[1] - block_size
        patch = original_frame[y:y + block_size, x:x + block_size]
        sampled_patches.append((patch, (y, x)))

    return sampled_patches


def merge_patches_into_image(patches, grid_size=7, block_size=32, image_size=(224, 224)):
    result_image = Image.new('RGB', image_size)
    for idx, (patch, _) in enumerate(patches):
        row = idx // grid_size
        col = idx % grid_size
        patch_image = Image.fromarray(patch)
        result_image.paste(patch_image, (col * block_size, row * block_size))

    return result_image


def process_video_frame_differences(video_name, frames, transform, resize=224):
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_name}")
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    print(f"正在处理视频 {video_name}, 视频长度: {video_length}, 帧率: {video_frame_rate}")
    transformed_video_global = torch.zeros([frames, 3, resize, resize])
    transformed_rs = torch.zeros([frames, 3, resize, resize])

    success, frame1 = cap.read()
    if not success:
        raise ValueError("无法读取第一帧")
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    last_frame = None

    for i in range(frames):
        frame_number = max(int(video_frame_rate / 2), 1) * i + max(1, int(video_frame_rate / 4))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame2 = cap.read()
        if success:
            frame_diff = calculate_frame_difference(frame1, frame2)
            original_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            sampled_patches = sample_patches_from_frame_difference(frame_diff, original_frame)
            result_image = merge_patches_into_image(sampled_patches)
            last_frame = result_image
            transformed_video_global[i] = transform(result_image)

            transformed_rs[i] = transform(Image.fromarray(original_frame).resize((resize, resize)))
            frame1 = frame2
        else:
            if last_frame is None:
                raise ValueError(f"无法读取帧 {i}, 且没有上一帧可用")
            print(f"无法读取帧 {i}, 使用上一帧代替")
            transformed_video_global[i] = transform(last_frame)
            transformed_rs[i] = transform(
                Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)).resize((resize, resize)))

    cap.release()
    return transformed_video_global, transformed_rs


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
        x= x.unsqueeze(0)
        y = y.unsqueeze(0)
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


def main(config):
    device = torch.device("cuda")
    model = ViT_32_Swin_Tiny_Fast_densenet121_Model(device)
    pretrained_weights_path = config.Model_weights_path
    if pretrained_weights_path:
        try:
            model.load_state_dict(torch.load(pretrained_weights_path, map_location=device, weights_only=True), strict=False)
            print(f"成功加载预训练权重: {pretrained_weights_path}")
        except Exception as e:
            print(f"加载预训练权重时出错: {e}")

    model.eval()
    model.float()
    model.to(device)
    # session_start_test = time.time()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        transformed_video_global, transformed_rs = process_video_frame_differences(config.videos_dir, config.frames, transform, config.resize)
        session_start_test = time.time()
        output = model(transformed_video_global.to(device), transformed_rs.to(device))
        print('得分为：', output)
        session_end_test = time.time()
        print('CostTime: {:.4f}'.format(session_end_test - session_start_test))
        flops, params = profile(model, inputs=(transformed_video_global.to(device), transformed_rs.to(device)))
        print(f"Total GFLOPs: {flops / 1e9}")
    except Exception as e:
        print(f"处理视频时出错: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', type=str, default=r'C:\Users\dell\Desktop\UGC\test_video\MP4\0001.mp4')
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--length_read', type=int, default=8)
    parser.add_argument('--grid_size', type=int, default=7)
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--Model_weights_path', type=str, default='PRE_LSVQ.pth')
    parser.add_argument('--resize', type=int, default=224)
    config = parser.parse_args()
    main(config)
