from pymatting import *
import numpy as np

scale = 1.0

image = load_image("../COCO_train2014_000000000081.jpg", "RGB", scale, "box")
trimap = load_image("../Background_Remvoal/trimap1.png", "GRAY", scale, "nearest")

# estimate alpha from image and trimap
alpha = estimate_alpha_cf(image, trimap)

# make gray background
background = np.zeros(image.shape)
background[:, :] = [0.5, 0.5, 0.5]

# estimate foreground from image and alpha
foreground = estimate_foreground_ml(image, alpha)

# blend foreground with background and alpha, less color bleeding
new_image = blend(foreground, background, alpha)
save_image('../new_image.jpg',new_image)
