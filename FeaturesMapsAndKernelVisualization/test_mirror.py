from mirror import mirror
from mirror.visualisations.web import *
from PIL import Image
from torchvision.models import resnet101, resnet18, vgg16, alexnet
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage

import numpy as np

import torch

from Models.rawConvNet import Model
from PrepareAndLoadData.load_prepared_dataset_in_dataloaders import load_dataloaders

# create a model
# Define Model
path_weights='../weights/TL_best_weights.pt'
path_bn_statistics="../weights/bn_statistics.pt"
model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35).cuda()
best_weights = torch.load(path_weights)
model.load_state_dict(best_weights)
list_dictionaries_bn_weights = torch.load(path_bn_statistics)
BN_weights = list_dictionaries_bn_weights[0]
model.load_state_dict(BN_weights, strict=False)
# open some images
path_dataset = '../Dataset/processed_dataset'
participants_dataloaders_train = load_dataloaders(path_dataset, batch_size=1, validation_cycle=None,
                                                  get_test_set=False, drop_last=False, shuffle=False)
for image, label in participants_dataloaders_train[0]:
    sample_image = image[0]
    break
print(np.shape(sample_image))

cat = Image.open("images/cat.jpg")
dog_and_cat = Image.open("images//dog_and_cat.jpg")
# resize the image and make it a tensor
to_input = Compose([ToPILImage(), Resize((224, 224)), ToTensor()])
# call mirror with the inputs and the model
mirror([sample_image], model, visualisations=[BackProp, GradCam, DeepDream])