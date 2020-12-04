import torch
import torchvision
import torchvision.io

import os
import numpy as np
from PIL import Image
from utils import Config
import cv2
from tqdm import tqdm

def download_videos():
    videos_list = open(Config['video_url_list'],'r').readlines()
    videos_list = [v.strip() for v in videos_list]
    print(videos_list)
    if not os.path.exists(Config['video_dataset_path']):
        os.makedirs(Config['video_dataset_path'])
    for i, video_url in enumerate(videos_list):
        print(video_url)
        video_path = os.path.join(Config['video_dataset_path'],'%d.mp4'%i)
        if os.path.exists(video_path):
            video_path = os.path.join(Config['video_dataset_path'],''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))+'.mp4')
        cmd = 'youtube-dl '+video_url+' -o '+video_path
        os.system(cmd)

def extract_frames():
    videos_list = os.listdir(Config['video_dataset_path'])
    # videos_list = [video for video in videos_list if video.endswith('.mp4')]
    if not os.path.exists(Config['image_dataset_path']):
        os.makedirs(Config['image_dataset_path'])
    for v_idx, video in enumerate(tqdm(videos_list)):
        # reader = torchvision.io.VideoReader(os.path.join(Config['video_dataset_path'],video),'video')
        cap = cv2.VideoCapture(os.path.join(Config['video_dataset_path'], video))
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is None or frame is None:
                break
            if (f_idx+1)%Config['skip_rate']==0:
                cv2.imwrite( os.path.join(Config['image_dataset_path'],
                                str(v_idx)+'_frame'+str(f_idx)+'.png'), frame)
                # torchvision.io.write_png(frame,
                #     os.path.join(Config['image_dataset_path'],
                #                 str(v_idx)+'_frame'+str(counter)+'.png'))
            f_idx+=1

if __name__=='__main__':
    if Config['mode']=='FULL' or Config['mode']=='VIDEO_DOWNLOAD':
        download_videos()
    if Config['mode']=='FULL' or Config['mode']=='FRAME_EXTRACT':
        extract_frames()
