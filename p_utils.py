import pandas as pd
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt



def image_preprocess(img):
    image=img.resize((90,108), resample = Image.LANCZOS)
    img = tf.constant(image, dtype=tf.float32)
    image=np.array(img/255)
    plt.imshow(image)
    #since it's greyscale and therefore doesn't have a 3 after the size,
    # (90,108,3) for shape in rgb it needs a one (90,108,1) for grey scale
    image=np.expand_dims(image,-1)
    # since it's a single img it still needs a batch size of one to be the right shape
    image=np.expand_dims(image,0)
    print(image.shape)
    return image