import numpy as np
from pspnet import Pspnet
from PIL import Image
import numpy as np
import os

class miou_Pspnet(Pspnet):
    def detect_image(self, image):
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        img, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        img = [np.array(img)/255]
        img = np.asarray(img)

        pr = self.model.predict(img)[0]
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0], self.model_image_size[1]])
        pr = pr[int((self.model_image_size[0]-nh)//2): int((self.model_image_size[0]-nw)//2+nh), int((self.model_image_size[1]-nw)//2): int((self.model_image_size[1]-nw)//2+nw)]


        image = Image.fromarray(np.uint8(pr)).resize((original_w, original_h))

        return image

pspnet = miou_Pspnet()

image_ids = open('val.txt', 'r').read().splitlines()

if not os.path.exists("./miou_pr_dir"):
    os.makedirs("./miou_pr_dir")

for image_id in image_ids:
    image_path = "./image/"+image_id+".jpg"
    image = Image.open(image_path)
    image = pspnet.detect_image(image)
    image.save("./miou_pr_dir/" + image_id + ".png")
    print(image_id, " done!")