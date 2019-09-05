#implemented from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division
import matplotlib.pyplot as plt
import torch, os, argparse, subprocess, random, time, copy, pdb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
#import torchvision
from torchvision import models, transforms
from PIL import Image
import cv2
import pandas as pd

#import matplotlib.pyplot as plt
#plt.ion() interactive mode

class CountingDataset(Dataset):

    def __init__(self, data_dict, transforms=None):
        self.data_list = list(data_dict.keys())
        self.data_dict = data_dict
        self.transforms = transforms

    def __getitem__(self, index):

        with open(self.data_list[index], 'rb') as f:
            img = Image.open(f).convert('RGB')

        return self.transforms(img), self.data_dict[self.data_list[index]]

    def __len__(self):
        return len(self.data_dict)

class FishCounter:

    def __init__(self, dataLoaderCommand, lossFunction = 'CCE', optimizer = 'adam', scheduler = 'step', modelDepth = '50', frozenFlag = False, saved = True, device = '0', lr = 0.0001):

        self.device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu") #default: '0'

        self.prepareData(dataLoaderCommand) #no default
        self.createModel(lossFunction, modelDepth, frozenFlag, saved) #default: 'CCE', '50', False, True
        self.setCriterion(lossFunction) #default: 'CCE'
        self.setOptimizer(optimizer, lr) #default: 'adam' , 1e-4
        self.setScheduler(scheduler) #default: 'step'

        
    def prepareData(self, command):
        commands = ['Train', 'UseCase']
        if command not in  commands:
            raise ValueException('dataLoaderCommand argument must be one of ' + ','.join(commands))

        if command == 'Train':
            self.dataLoaderTrain()

        if command == 'UseCase':
            self.dataLoaderUseCase()
            
            
    def dataLoaderTrain(self):

        subprocess.call(['rclone', 'copy', 'cichlidVideo:McGrath/Apps/CichlidPiData/__Counting/', 'CountingData']) #works on server

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

        image_data = {}
        image_data['val'] = {}
        image_data['train'] = {}

        
        for project in [x for x in os.listdir('CountingData/') if x[0] != '.']:
            for video in [x for x in os.listdir('CountingData/' + project) if x[0] != '.']:
                for label in [x for x in os.listdir('CountingData/' + project + '/' + video) if x[0] != '.']:
                    if label != 'p':
                        for videofile in [x for x in os.listdir('CountingData/' + project + '/' + video + '/' + label) if x[-3:] == 'jpg']:
                            if random.randint(0,4) == 0: #20:80 val:train
                                image_data['val']['CountingData/' + project + '/' + video + '/' + label + '/' + videofile] = int(label)
                            else:
                                image_data['train']['CountingData/' + project + '/' + video + '/' + label + '/' + videofile] = int(label)
        

        self.dataloaders = {x: DataLoader(CountingDataset(image_data[x], transforms = data_transforms[x]), batch_size=32, shuffle=True, num_workers=4, sampler=None) for x in ['train', 'val']}


    def dataLoaderUseCase(self):

        data_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
            ])

        image_data = {}


        for video in os.listdir(video_dir):
            vidcap = cv2.VideoCapture(video_dir + '/' + video)
            success = False
            while not success:
                success, image = vidcap.read()


            cv2.imshow('Identify the parts of the frame that include tray to analyze', image)
            tray_r = cv2.selectROI('Identify the parts of the frame that include tray to analyze', image, fromCenter = False)
            tray_r = tuple([int(x) for x in tray_r]) # sometimes a float is returned
            self.tray_r = [tray_r[1],tray_r[0],tray_r[1] + tray_r[3], tray_r[0] + tray_r[2]] # (x0,y0,xf,yf)

            # if bounding box is close to the edge just set it as the edge
            if self.tray_r[0] < 50:
                self.tray_r[0] = 0
            if self.tray_r[1] < 50:
                self.tray_r[1] = 0
            if image.shape[0] - self.tray_r[2]  < 50:
                self.tray_r[2] = image.shape[0]
            if image.shape[1] - self.tray_r[3]  < 50:
                self.tray_r[3] = image.shape[1]

            vidcap.release()
            # Destroy windows (running it 3 times helps for some reason)
            for i in range(3):
                cv2.destroyAllWindows()
                cv2.waitKey(1)

            count = 0
            #pdb.set_trace()
            success = True
            while success:
                #cv2.imwrite(os.path.join(test_dir, "frame%d.jpg" % count), image)
                print("Read frame #%d: " % count, success)
                dataArray = image

                #Set data outside of tray to np.nan
                dataArray[:,:self.tray_r[0],:] = np.nan
                dataArray[:,self.tray_r[2]:,:] = np.nan
                dataArray[:,:,:self.tray_r[1]] = np.nan
                dataArray[:,:,self.tray_r[3]:] = np.nan

                PImage = Image.fromarray(dataArray)
                PImage.save(os.path.join(test_dir, "frame%d.jpg" % count))

                image_data[os.path.join(test_dir, "frame%d.jpg" % count)] = int(count)


                success, image = vidcap.read()
                count += 1

        ###fill image_data with video frame directories & time points
        self.testDataloader = DataLoader(CountingDataset(image_data, transforms = data_transforms), batch_size = 32, shuffle = False, num_workers = 4, sampler = None)

        
    def setCriterion(self, lossFunction):
        lossFunctions = ['L1', 'L2', "CCE"]
        if lossFunction not in lossFunctions:
            raise ValueException('command argument must be one of ' + ','.join(lossFunctions))

        if lossFunction == 'L1':
            self.criterion = nn.L1Loss()
        elif lossFunction == 'L2':
            self.criterion = nn.MSELoss()
        elif lossFunction == 'CE':
            cweights = [0.6035, 0.6137, 0.8485, 0.9499, 10, 100] #should probably raise 3 to 0.9499, initially 4:0.9851
            class_weights = torch.FloatTensor(cweights).to(device)
            self.criterion = nn.CrossEntropyLoss(weight = class_weights)

            
    def setOptimizer(self, optimizer, lr = 0.0001):
        optimizers = ['adam', 'sgd']
        if optimizer not in optimizers:
            raise ValueException('optimizer argument must be one of ' + ','.join(optimizers))

        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

            
    def setScheduler(self, scheduler, **kwargs):
        schedulers = ['none', 'step', 'cycle']
        if scheduler not in schedulers:
            raise ValueException('scheduler argument must be one of ' + ','.join(schedulers))
        # Decay LR by a factor of 0.1 every 7 epoch_loss

        if scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 7, gamma = 0.1)

            
    def createModel(self, lossFunction, depth, frozenFlag, saved):

        lossFunctions = ['L1', 'L2', 'CCE']
        if lossFunction not in lossFunctions:
            raise ValueException('lossFunction argument must be one of ' + ','.join(lossFunctions))

        depth = str(depth)
        depths = ['10', '18', '34', '50']
        if depth not in depths:
            raise ValueException('modelDepth argument must be one of ' + ','.join(depths))

        if depth == '10':
            model = models.resnet10(pretrained=True)
        if depth == '18':
            model = models.resnet18(pretrained=True)
        if depth == '34':
            model = models.resnet34(pretrained=True)
        if depth == '50':
            model = models.resnet50(pretrained=True)

        num_ftrs = model.fc.in_features

        if frozenFlag:
            for param in model.parameters():
                param.requires_grad = False

        if lossFunction == 'CCE':
            model.fc = nn.Linear(num_ftrs, 6)
        else:
            model.fc = nn.Linear(num_ftrs, 1)

        if saved:
            old_save_path = os.path.join(os.getcwd(), 'resnetCountSRG.pth')
            model = torch.load(old_save_path)
            #self.model = torch.load_state_dict(old_save_path)

        self.model = model


    def trainModel(self, num_epochs=55):

        since = time.time()
        self.model = self.model.to(self.device)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for e in range(num_epochs):
            print('Epoch {}/{}'.format(e, num_epochs - 1))
            print('-' * 10)

            #training/validation
            for phase in ['train', 'val']:
                if phase == 'train':

                    self.model.train() #set model to training mode
                else:
                    self.model.eval() #set model to evaluation mode

                running_loss, running_corrects = 0.0, 0

                #iterate over data
                for inputs, labels in self.dataloaders[phase]:

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    #pdb.set_trace()

                    #zero parameter gradients
                    self.optimizer.zero_grad()

                    #forward pass
                    #track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)

                        if type(self.criterion) == torch.nn.modules.CrossEntropyLoss:
                            _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, labels)
                        else:
                            preds = outputs.int()[:,-1].type(torch.int64) #don't know if this works yet
                            loss = self.criterion(outputs[:,-1], labels.float())

                        #backward pass / optimization only in training
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    #stats
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / len(self.dataloaders[phase])
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                #deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print() #empty line

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val accuracy: {:.4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)

        self.model = model

        confusion_matrix = torch.zeros(6, 6)
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
        
        print('Confusion Matrix:')
        print(confusion_matrix)

        new_save_path = os.path.join(os.getcwd(), 'fishCounter.pth')
        torch.save(model.state_dict(), new_save_path)

        
    def useModel(self, path):

        graphData = pd.Series([])

        for inputs, labels in self.testDataloader:

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            #pdb.set_trace()

            outputs = self.model(inputs)

            graphData.append(torch.max(outputs, 1))

        histGraph = graphData.hist()
        fig = histGraph.get_figure()
        fig.savefig(os.getcwd() + '/figure.pdf')

        graphData.plot.line(x = 'Time', y = 'Fish Count')


        
#train/evaluate
#__init__(self, dataLoaderCommand, lossFunction = 'CCE', optimizer = 'adam', scheduler = 'step', modelDepth = '50', frozenFlag = False, saved = True, device = '1', lr = 0.0001, )

video_dir = os.path.join(os.getcwd(), 'TestVideos')
test_dir = os.path.join(os.getcwd(), 'TestData')

FC = FishCounter('UseCase')
FC.useModel(test_dir)

#FC.trainModel(num_epochs = 100)
