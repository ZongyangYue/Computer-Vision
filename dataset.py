# Dataset class
# all subclasses should overwrite __getitem__ method
# optionally overwrite __len__ method
# acquire the data and the corresponding labels
# __getitem__(self, index): get a single data and its label
# __len__: the data size
# __add__(self, other): 
# 
from torch.utils.data import Dataset
from PIL import Image # opens an image through a path
import cv2 #opencv opens an image through a path
import os


# The data directory is in this way:
# ./data
# ------ants/ (folder name is also the label name)
#       ----img1
#       ----img2
#       ----img3
# ------bees/
#       ----img1
#       ----img2
#       ----img3
class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        #if root_dir is "dataset/train"
        # label_dir is "ants"
        #then self.path is "dataset/train/path"
        
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir) 
        self.image_path_list = os.listdir(self.path)
        

    def __getitem__(self, index):
        image_name = self.image_path_list[index]
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        image = Image.open(image_item_path)
        label = self.label_dir
        return image, label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
ants_label_dir = "ants"
ants_data_set = MyData(root_dir, ants_label_dir)
bees_label_dir = "bees"
bees_data_set = MyData(root_dir, bees_label_dir)

# aggregate two datasets to form the train dataset
train_dataset = ants_data_set + bees_data_set 

print(len(ants_data_set)) #124
print(len(bees_data_set)) #121
print(len(train_dataset)) # 245 = 124 + 121
# the __len__ will be the length of ants dataset = number of images in ants/

    

        