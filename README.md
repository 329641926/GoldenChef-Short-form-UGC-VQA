### Train the model
- Attention
  **First of all, you should pip install CLIP-main with commands below**
  ```python
  cd CLIP-main
  pip install .
  ```
- Extract the images:
  
  ```python
  python Frame_Difference_Patches.py      \
  --dataset /mnt/f/KVQ/val/MP4/           \    # location of your dataset
  --save_path /mnt/f/KVQ/                 \    # while done, then get a Frame_Difference_Patches dir
  --csv_path /mnt/f/KVQ/val/truth.csv     \
  --video_query_symbol filename           \    # title of the video name in the .csv file   
  >> Frame_Difference_Patches.log
  ```

- Train the model:
  
  ```python
  python train.py \
  --train_dir train_video\
  --train_datainfo data/train_data.csv \
  --val_dir val_video \
  --val_datainfo data/val_data.csv \
  --test_dir test_video\
  --test_datainfo data/test_data.csv \
  --frames 12 \
  --conv_base_lr 2e-5 \
  --decay_ratio 0.9 \
  --decay_interval 2 \
  --print_samples 1000 \
  --train_batch_size 6 \
  --num_workers 8 \
  --resize 224\
  --epochs 30 \
  --sample_rate 1 \
  --pretrained_weights_path "" \
  >> logs/train.log
  ```

### Test

```python
python test.py \
--videos_dir val_video/MP4    \
--datainfo data/val_data.csv    \
--frames 12    \
--Model_weights_path weight/PRE_LSVQ.pth    \
--resize 224    \
--num_workers 8
```

### TEST_ONE_VIDEO_RUNNINGTIME

```python
python test_one_video.py    \
--videos_dir /mnt/cUsers/dell/Desktop/UGC/test_video/MP4/0001.mp4    \
--frames 12    \
--Model_weights_path PRE_LSVQ.pth \
```


