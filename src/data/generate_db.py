import torch
import torchvision

import os
import numpy as np
from PIL import Image
from utils import Config

def download_videos():
    videos_list = open(Config['video_url_list'],'r').readlines()
    videos_list = [v.strip() for v in videos_list]
    if not os.path.exists(Config['video_dataset_path']):
        os.makedirs(Config['video_dataset_path'])
    for i, video_url in enumerate(videos_list):
        video_path = os.path.join(Config['video_dataset_path'],'%d.mp4'%i)
        if os.path.exists(video_path):
            video_path = os.path.join(Config['video_dataset_path'],''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))+'.mp4')
        cmd = 'youtube-dl '+video_url+' -o '+video_path
        os.system(cmd)

def extract_frames():
    videos_list = os.listdir(Config['video_dataset_path'])
    videos_list = [video for video in videos_list if video.endswith('.mp4')]
    if not os.path.exists(Config['image_dataset_path']):
        os.makedirs(Config['image_dataset_path'])
    for v_idx, video in enumerate(videos_list):
        reader = torchvision.io.VideoReader(os.path.join(Config['video_dataset_path'],video), 'video')
        for counter, frame in enumerate(reader):
            if (counter+1)%Config['skip_rate']:
                torchvision.io.write_png(frame,
                    os.path.join(Config['image_dataset_path'],
                                str(v_idx)+'_frame'+str(counter)+'.png'))


if __name__=='__main__':
    if Config['mode']=='FULL' or Config['mode']=='VIDEO_DOWNLOAD':
        download_videos()
    if Config['mode']=='FULL' or Config['mode']=='FRAME_EXTRACT':
        extract_frames()
