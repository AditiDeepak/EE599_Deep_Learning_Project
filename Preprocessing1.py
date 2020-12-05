#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip3 install PyMatting')


# In[5]:


get_ipython().system('pip3 install opencv-contrib-python')


# In[82]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.image as mpimg
# from pymatting import *
import os
from glob import glob


# In[83]:


#Performing filtering operation
def Salt_and_pepper_noise(image):
    count = 0
    lastMedian = image
    median = cv2.medianBlur(image, 3)
    while not np.array_equal(lastMedian, median):
        zeroed = np.invert(np.logical_and(median, image))
        image[zeroed] = 0

        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(image, 3)
    return image

#find the significant contour
def Contour(image):
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
            
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])
    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour


# In[90]:


input_images = glob('./inputs/*.jpg')
# for i in range(len(input_images)):
input_image = cv2.imread(input_images[3])
input_image = input_image[:,:,::-1]
#perform gaussion blur
blur = cv2.GaussianBlur(input_image, (5, 5), 0)
blur = blur.astype(np.float32) / 255.0
#use the model.yml file to perform edge detection (pre-trained)
edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
edges = edgeDetector.detectEdges(blur) * 255.0
# cv2.imwrite('edge-raw.jpg', edges)

edges_8u = np.asarray(edges, np.uint8)
Salt_and_pepper_noise(edges_8u)
# cv2.imwrite('edge.jpg', edges_8u)

contour = Contour(edges_8u)

# Draw the contour on the original image
contourImg = np.copy(input_image)
cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

#Generate trimap
mask = np.zeros_like(edges_8u)
cv2.fillPoly(mask, [contour], 255)

# calculate sure foreground area by dilating the mask
mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

trimap = np.copy(mask)
trimap[mask == 0] = cv2.GC_BGD
trimap[mask == 255] = cv2.GC_PR_BGD
trimap[mapFg == 255] = cv2.GC_FGD

# visualize trimap
trimap_print = np.copy(trimap)
trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
trimap_print[trimap_print == cv2.GC_FGD] = 255

mask_path = "./mask_images/"
trimap_path = "./trimap_images/"
target_path = "./target_images/"

try:
    os.stat(trimap_path)
except:
    os.mkdir(trimap_path)

try:
    os.stat(target_path)
except:
    os.mkdir(target_path)

try:
    os.stat(mask_path)
except:
    os.mkdir(mask_path)  

cv2.imwrite(trimap_path + 'trimap_' + str(i) +'.png', trimap_print)

# run grabcut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
cv2.grabCut(input_image, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

# create mask again
mask2 = np.where((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),255,0).astype('uint8')
cv2.imwrite(mask_path + 'mask_' + str(i) +'.png', mask2)

# estimate alpha from image and trimap
alpha = estimate_alpha_cf(input_image, trimap)
# alpha = mask2

# make gray background
background_image = cv2.imread(random.choice(glob('./background/*jpg')), cv2.IMREAD_COLOR)
background_image = cv2.resize(background_image, (640, 494), interpolation = cv2.INTER_AREA)
# background_image =  (1 - alpha[:,:,np.newaxis]) * background_image 
# cv2.imwrite( 'back.png', background_image)

# estimate foreground from image and alpha
# foreground = input_image[:,:,::-1] * alpha[:,:,np.newaxis]
# cv2.imwrite( 'front.png', foreground)
foreground = estimate_foreground_ml(input_image,alpha)

# blend foreground with background and alpha, less color bleeding
out_image = blend(foreground, background_image,alpha)
cv2.imwrite( target_path + 'target_' + str(i) +'.png', out_image)


# In[69]:


input_image  = input_image[:,:,::-1]*alpha[:,:,np.newaxis]


# In[65]:


cv2.imwrite( 'foreground.png', input_image)


# In[64]:


background_image = cv2.resize(background_image, (640, 494), interpolation = cv2.INTER_AREA)


# In[73]:


background_image = background_image * (1 - alpha[:,:,np.newaxis])


# In[71]:


cv2.imwrite( 'background.png', background_image)


# In[72]:


out_image = cv2.add(foreground, background_image)


# In[ ]:


cv2.imwrite( 'background.png', background_image)


# In[89]:


alpha = mask2
foreground = input_image[:,:,::-1] * alpha[:,:,np.newaxis]
cv2.imwrite( 'foreground1.png', input_image)


# In[ ]:




