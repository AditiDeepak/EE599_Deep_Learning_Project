# EE599_Deep_Learning_Project
GANs in Video Conferencing

Files for SRGAN:
data.py - Initial dataloader design for SRGAN returns (lr_image, hr_image)

model_edsr.py - Model design adapted from EDSRGAN and Fast-SRGAN with content_loss and adversarial_loss functions EDSR_Generator and Discriminator classes are the generator and discriminator respectively

model.py - Modified architecture for reduced parameters Generator and Discriminator classes are the main parts feat_extractor is a Resnet50

run_inference.py - skeleton code for running inference on the trained model

train.py - final training routine for our modified architecture

trainer.py - training routine for edsrgan/fastsrgan models from model_edsr.py

utils.py - Config and metric classes

SRGAN takes as input normalized image (1,3,240,320) image and gives as output (1,3,480,640)

Files for data:
generate_db.py - performs video to frame conversion, youtube video downloads

green_screen_removal.py - Code used for testing purposes, didn't help much

gsr_test.py - Code used for calibrating green screen range

gsr.py - Another code for doing green screen removal

utils.py - Config dictionary

videos*.txt - list of youtube urls to be downloaded

Files for BGRSRGAN:

model.py - Generator class as shown in report, Discriminator Resnet18, Feature extractor used Resnet34, Resnet50

run_inference.py - Performs inference on green screen video input for testing purposes

train.py - Final training routine for the BGRSRGAN architecture

utils.py - Config dictionary and PSNR metrics, SSIM was changed to skimage toolkit

BGRSRGAN takes as input normalized image with original background, target background concatenated channelwise (1,6,240,320) and returns two output images i) background replaced low resolution image (1,3,240,320) and ii) background replaced super resolution image (1,3,480,640)
