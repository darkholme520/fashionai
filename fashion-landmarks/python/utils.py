from PIL import Image
import numpy as np
def imread(img_name):
    im_np = np.array(Image.open(img_name))
    return im_np