from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

# store events in the "./logs" folder
writer = SummaryWriter("logs")

image_path = "dataset/train/ants/0013025.jpg"
img = Image.open(image_path)
img = np.array(img)

# add_image takes in tensor or np array image format
# default dataformats is CHW, channel, height, width
# we can set dataformats = "HWC" if we have (100, 100, 3)
# add_image("title", numpyarray/tensor, step, dataformats)
# see how images change at different steps
writer.add_image("title", img, 1, dataformats='HWC')


# tag: title of the plot, scalar_value: value to save (y-axis)
# global step (x-axis)
for i in range(100):
    #writer.add_scalar("y=x", i, i) 
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
