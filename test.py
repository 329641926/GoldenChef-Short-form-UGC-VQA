import os
import cv2
from torch.utils import data
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
import time
import numpy as np
import argparse
import UGCModel
from utils import performance_fit


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
            # print(f"无法读取帧 {i}, 使用上一帧代替")
            transformed_video_global[i] = transform(last_frame)
            transformed_rs[i] = transform(
                Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)).resize((resize, resize)))

    cap.release()
    return transformed_video_global, transformed_rs


class VideoDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize, frames):
        super(VideoDataset, self).__init__()

        dataInfo = pd.read_csv(filename_path)
        self.video_names = dataInfo['filename'].tolist()
        self.score = dataInfo['score'].tolist()
        self.resize = resize
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.frames = frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = str(self.video_names[idx]).split('/')[-1]
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))
        video_path = os.path.join(self.videos_dir, video_name)
        transformed_video_global, transformed_rs = process_video_frame_differences(video_path, self.frames,
                                                                                   self.transform)

        return transformed_video_global, transformed_rs, video_name, video_score


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UGCModel.ViT_32_Swin_Tiny_Fast_densenet121_Model()

    pretrained_weights_path = config.Model_weights_path
    if pretrained_weights_path:
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device, weights_only=True), strict=True)
        print(f"成功加载预训练权重: {pretrained_weights_path}")

    model.to(device)
    model.float()

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testset = VideoDataset(config.videos_dir, config.datainfo, transformations, config.resize, config.frames)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()
        session_start_test = time.time()
        label = np.zeros([len(testset)])
        y_output = np.zeros([len(testset)])
        names = []

        for i, (video, rs, name, score) in enumerate(test_loader):
            video = video.to(device)
            rs = rs.to(device)
            names.append('test/' + name[0])
            label[i] = score.item()
            outputs = model(video, rs)
            y_output[i] = outputs.item()

        session_end_test = time.time()
        val_PLCC, val_SRCC, val_KRCC, val_RMSE = performance_fit(label, y_output)
        print(
            'completed. The result : SRCC: {:.4f}, KRCC: {:.4f}, '
            'PLCC: {:.4f}, and RMSE: {:.4f}'.format(val_SRCC, val_KRCC, val_PLCC, val_RMSE))

        data = {
            'filename': names,
            'score': y_output
        }

        df = pd.DataFrame(data)
        df.to_csv('prediction.csv', index=False)
        print(f'CostTime: {session_end_test - session_start_test:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--videos_dir', type=str, default='val_video/MP4')
    parser.add_argument('--datainfo', type=str, default='data/val_data.csv')
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--Model_weights_path', type=str, default='weight/PRE_LSVQ.pth', help='模型权重路径')
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=8)
    config = parser.parse_args()
    main(config)
