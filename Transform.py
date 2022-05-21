from torchvision import transforms
import cv2
from PIL import Image

# cv2.imread returns a numpy.array type
img_path = "dataset/train/ants/0013025.jpg"
cv_img = cv2.imread(img_path)

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(cv_img)

# Normalize: input[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_normed = trans_norm(img_tensor)

#Resize: input is a PIL image
img = Image.open(img_path)
trans_resize = transforms.Resize([512, 512])
img_resize = trans_resize(img)

#Compose: input is a list of transform objects
#note that the previous transform's output should be 
#suitable format for next transform's input
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_norm])
img_resize_2 = trans_compose(img)

#when you do not know return type, use print(), print(type())
# RandomCrop, input is a PIL image
trans_random_crop = transforms.RandomCrop([512, 512])

for i in range(10):
    img_crop = trans_random_crop(img)


