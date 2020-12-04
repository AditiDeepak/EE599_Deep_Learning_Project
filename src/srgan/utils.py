Config={}

Config['debug']=True
Config['use_cuda']=False
Config['train_set_path']='/home/adityan/EE599_Deep_Learning_Project/src/data/youtube_vid_dataset'
# Config['test_set_path']='/home/adityan/EE599_Deep_Learning_Project/src/data/images'
Config['checkpoint_path']='/home/adityan/EE599_Deep_Learning_Project/src/data/checkpoints/srgan'

Config['scale']=2
Config['n_colors']=3
Config['n_resblocks']=6
Config['n_feats']=64
Config['epochs']=300

Config['batch_size']=2
Config['num_workers']=2
Config['img_size']=(480,640)

Config['generator_lr']=1e-4
Config['discriminator_lr']=1e-4
Config['optimizer']='Adam'
Config['skip_threshold']=1e8
Config['tensorboard_log']=True

Config['generator_checkpoint']=None
Config['discriminator_checkpoint']=None
