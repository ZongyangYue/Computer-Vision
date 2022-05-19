#-------------------------------#
#      predict single image     #
#-------------------------------#
from unet import Unet
from PIL import Image

unet = Unet()

while True:
    img = input('Input image filename: ')
    try:
        image = Image.open(img)
    except:
        print("Open Error! Try Again!")
        continue
    else:
        r_image = unet.detect_image(image)
        r_image.show()