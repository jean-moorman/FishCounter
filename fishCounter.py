#implemented from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
### CrossEntropyLoss w/ unfrozen resnet50
#achieved 94% accuracy with 10000 frames

from __future__ import print_function, division
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import WeightedRandomSampler
#from sampler import ImbalancedDatasetSampler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import time
import copy
import pdb
#from PIL import Image

plt.ion() #interactive mode on

data_transforms = {
    'train': transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.9,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #this should probably be customized
    ]),
    'val': transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #this too
    ])
}

data_dir = '/home/username/Desktop/Counting_Data'

#mis_dir = '/home/username/Desktop/Mislabeled_Counts'

cweights = [0.6035, 0.6137, 0.8485, 0.9499, 0.9851, 1] #class weights for samples 3942:3841:1506:498:148:8

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4, sampler=None) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=55):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for e in range(num_epochs):
        print('Epoch {}/{}'.format(e, num_epochs - 1))
        print('-' * 10)

        #training/validation
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train() #set model to training mode
            else:
                model.eval() #set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            #iterate over data
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                #pdb.set_trace()

                #zero parameter gradients
                optimizer.zero_grad()

                #forward pass
                #track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #output = outputs[:, -1] ###only use for L1 or MSELoss
                    loss = criterion(outputs, labels) #should be labels.float() for L1 or MSELoss

                    #backward pass / optimization only in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            #deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print() #empty line

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model

# def valid_imshow_data(data):
#     data = np.asarray(data)
#     if data.ndim == 2:
#         return True
#     elif data.ndim == 3:
#         if 3 <= data.shape[2] <= 4:
#             return True
#         else:
#             print('The "data" has 3 dimensions but the last dimension '
#                   'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
#                   ''.format(data.shape[2]))
#             return False
#     else:
#         print('To visualize an image the data must be 2 dimensional or '
#               '3 dimensional, not "{}".'
#               ''.format(data.ndim))
#         return False

'''
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)


            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

                dataOut = (inputs.cpu().data[j])
                plt.imshow(dataOut)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)
'''


model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features

#freezes all except last layer
'''
for param in model_ft.parameters():
    param.requires_grad = False
'''
model_ft.fc = nn.Linear(num_ftrs, 6)

#saving/loading state_dict
#save_path2 = '/home/username/Desktop/resnet_state.pth'
### model.load_state_dict(torch.load(save_path2))

#saving/loading state
save_path = '/home/jmoorman9/Desktop/resnetC.pth'
#comment out the next line to start from scratch
model_ft = torch.load(save_path)

model_ft = model_ft.to(device)

class_weights = torch.FloatTensor(cweights).to(device) #only for Cross Entropy Loss

criterion = nn.CrossEntropyLoss(weight = class_weights)
#criterion = nn.MSELoss()
#criterion = nn.L1Loss()

# Optimizers:
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 7 epoch_loss
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#train/evaluate
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs = 100)

confusion_matrix = torch.zeros(6, 6)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)

#torch.save(model_ft.state_dict(), save_path2)

torch.save(model_ft, save_path)

### visualize_model(model, save_path) <--- doesn't work yet

