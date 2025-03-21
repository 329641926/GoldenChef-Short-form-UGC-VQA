import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import numpy as np


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
        video_name_str = video_name[:-4]
        # video_name_str = str(self.video_names[idx])
        video_score = torch.FloatTensor(np.array(float(self.score[idx]/20)))
        # print(f"正在读取视频 {video_name_str}，分数 {video_score}")
        global_patch_path = os.path.join(self.videos_dir, 'Frame_Difference_Patches', video_name_str)
        frames_path = os.path.join(self.videos_dir, 'frames', video_name_str)
        video_length_read = self.frames
        transformed_video_global = torch.zeros([video_length_read, 3, self.resize, self.resize])
        transformed_rs = torch.zeros([video_length_read, 3, self.resize, self.resize])
        last_frame = None
        last_frame_path = None
        for i in range(video_length_read):
            frames = os.path.join(frames_path, '{:04d}'.format(int(i)) + '.png')
            try:
                frame = Image.open(frames)
                frame = frame.convert('RGB')
                frame = frame.resize((224, 224))
                frame = self.transform(frame)
                transformed_rs[i] = frame
                last_frame = frame
            except (FileNotFoundError, IOError) as e:
                # print(f"无法读取帧 {i}，使用最后一帧补充。")
                try:
                    transformed_rs[i] = last_frame
                except Exception as e:
                    print(f"无法读取最后一帧，错误信息：{e}，路径：{frames_path}")

            global_patches = os.path.join(global_patch_path, '{:04d}'.format(int(i)) + '_patch.png')
            try:
                global_frame = Image.open(global_patches)
                global_frame = global_frame.convert('RGB')
                global_frame = self.transform(global_frame)
                transformed_video_global[i] = global_frame
                last_frame_path = global_frame
            except (FileNotFoundError, IOError) as e:
                # print(f"无法读取帧 {i}，使用最后一帧补充。")
                try:
                    transformed_video_global[i] = last_frame_path
                except Exception as e:
                    print(f"无法读取最后一帧，错误信息：{e}，路径：{last_frame_path}")

        return transformed_video_global, transformed_rs, video_score, video_name_str
