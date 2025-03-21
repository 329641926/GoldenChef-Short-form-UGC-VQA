import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

import argparse


def calculate_frame_difference(frame1, frame2):
    # 计算帧差
    diff = cv2.absdiff(frame1, frame2)
    # 转换为灰度图
    # diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return diff


def sample_patches_from_frame_difference(frame_diff, original_frame, grid_size=7, block_size=32):
    grid_height = frame_diff.shape[0] // grid_size
    grid_width = frame_diff.shape[1] // grid_size
    blocks = []

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

                    block_start_y = m
                    block_end_y = min(m + block_size, grid_height)

                    block_start_x = n
                    block_end_x = min(n + block_size, grid_width)

                    small_block = grid_region[block_start_y:block_end_y, block_start_x:block_end_x]
                    s = np.sum(small_block)

                    if s > max_s:
                        max_s = s
                        max_position = (start_y + block_start_y, start_x + block_start_x)

            blocks.append(max_position)

    selected_positions = blocks

    sampled_patches = []
    for pos in selected_positions:
        y, x = pos
        if y + block_size > original_frame.shape[0]:
            y = original_frame.shape[0] - block_size
        if x + block_size > original_frame.shape[1]:
            x = original_frame.shape[1] - block_size
        patch = original_frame[y:y + block_size, x:x + block_size]
        sampled_patches.append((patch, pos))

    return sampled_patches


def merge_patches_into_image(patches, grid_size=7, block_size=32, image_size=(224, 224)):
    result_image = Image.new('RGB', image_size)

    for idx, (patch, _) in enumerate(patches):
        row = idx // grid_size
        col = idx % grid_size
        patch_image = Image.fromarray(patch)
        result_image.paste(patch_image, (col * block_size, row * block_size))

    return result_image


def process_video_frame_differences(video_name, video_number_min, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_dir_frames = os.path.join(output_dir.replace('Frame_Difference_Patches', 'frames'))
    os.makedirs(output_dir_frames, exist_ok=True)

    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        raise ValueError("not opened")
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    print(f" Processing {video_name}, Video length: {video_length}, frame rate: {video_frame_rate}")

    success, frame1 = cap.read()
    if not success:
        raise ValueError("Unable to read the first frame")
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    for i in range(0, video_number_min):
        frame_number = max(int(video_frame_rate / 2), 1) * i + max(1, int(video_frame_rate / 4))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame2 = cap.read()
        if success:
            frame_diff = calculate_frame_difference(frame1, frame2)
            # frame2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            original_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            sampled_patches = sample_patches_from_frame_difference(frame_diff, original_frame)
            result_image = merge_patches_into_image(sampled_patches)
            output_path = os.path.join(output_dir, f'{i:04d}_patch.png')
            output_frames = os.path.join(output_dir_frames, f'{i:04d}.png')
            result_image.save(output_path)

            read_frame = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            read_frame.save(output_frames)
            frame1 = frame2

    cap.release()
    return


def process_video(video_name_str):
    num_samples = 16
    video_name_str = video_name_str.split('/')[-1][:-4]
    # video_name_str = video_name_str
    video_path = os.path.join(config.dataset, str(video_name_str) + '.mp4')
    output_dir_frame_diff = os.path.join(config.save_path, 'Frame_Difference_Patches', str(video_name_str))
    process_video_frame_differences(video_path, num_samples, output_dir_frame_diff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=r"F:\KVQ\val\MP4")
    parser.add_argument("--save_path", type=str, default=r'F:\KVQ\val')
    parser.add_argument("--csv_path", type=str, default=r'F:\KVQ\val\truth.csv')
    parser.add_argument("--video_query_symbol", type=str, default=r"filename")

    config = parser.parse_args()

    dataInfo = pd.read_csv(config.csv_path)
    video_names = dataInfo[config.video_query_symbol].tolist()
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_video, video_names)
