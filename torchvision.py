import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=True)


# Before setting transform=dataset_transform
print(test_set[0]) # a PIL object

img, target = test_set[0] # img is a PIL object, target a number 3
print(test_set.classes) # ['airplane', 'automobile', 'bird', 'cat', 'deer'. 'frog', '
                        # horse', 'ship', 'truck']

# After setting transform=dataset_transform
# test_set[0] is a tensor