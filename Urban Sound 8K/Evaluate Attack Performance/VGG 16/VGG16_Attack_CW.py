import torch.optim as optim # Import optim 
import torchvision # Import torchvision 
import h5py #import h5py
from torch.utils.data import dataset # import dataset
from ESC10Customdataset_Train_FB import LAEData_Train # Call Customdataloader to read Training Data
from ESC10Customdataset_Test_FB import LAEData_Test # Call Customdataloader to read Test Data
from torch.utils.data import DataLoader # Import Dataloader 
import torchvision.transforms as transforms # Import Transform 
import pandas as pd # Import Pnadas 
import torch # Import Torch 
import torch.nn as nn # Import NN module from Torch 
from torchvision.transforms import transforms# Import transform module from torchvision 
from torch.utils.data import DataLoader # Import dataloader from torch 
from torch.optim import Adam # import optimizer module from torch 
from torch.autograd import Variable # Import autograd from torch 
import numpy as np # Import numpy module 
import torchvision.datasets as datasets #Import dataset from torch 
#from Attention import CAM_Module # import channel attention module
#from Attention import SA_Module # Import Self attention module
from torch import optim, cuda # import optimizer  and CUDA
import random # import random 
import torch.nn.functional as F # Import nn.functional 
import time # import time 
import sys # Import System 
import os # Import OS
from pytorchtools import EarlyStopping
from torchvision import models
import warnings
SEED = 1234 # Initialize seed 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda') # Define device type 
num_classes=10 # Define Number of classes 
in_channel=1   # Define Number of Input Channels 
learning_rate=2e-5 # Define Learning rate 
batch_size=4 # Define Batch Size 
EPOCHS =1000   # Define maximum Number of Epochs
FC_Size=512
SFC_Size=512
Temp=3
alpha=0.7
N_models=6
warnings.filterwarnings("ignore")
torch.backends.cudnn.allow_tf32 = True
train_transformations = transforms.Compose([ # Training Transform 
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
test_transformations = transforms.Compose([ # Test Transform 
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
train_dataset=LAEData_Train(transform=train_transformations) # Create tensor of training data 
Test_Dataset=LAEData_Test(transform=test_transformations)# Create tensor of test dataset 
train_size = int(0.7 * len(train_dataset)) # Compute size of training data using (70% As Training and 30% As Validation)
valid_size = len(train_dataset) - train_size # Compute size of validation data using (70% As Training and 30% As Validation)
Train_Dataset,Valid_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size]) # Training and Validation Data After (70%-30%)Data Split 
#train_set,test_set=torch.utils.data.random_split(dataset,[6000,2639])
#Labels=pd.read_csv("Devlopment.csv")
train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True) # Create Training Dataloader 
valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=False)# Create Validation Dataloader 
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False) # Create Test Dataloader 
class Teacher(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Teacher, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.vgg16(pretrained=True).children())[:-1])
        self.features=Pre_Trained_Layers
        #print(self.features)
        self.features[0][0]=nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1, 1), padding=(1, 1), bias=False)
        self.fc1=nn.Linear(25088,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,10)
    def forward(self,image):
        x1 = self.features(image)
        x1=x1.view(x1.shape[0],-1)
        x2=self.fc1(x1)
        x3=self.fc2(x2)
        x4=self.fc3(x3)
        return x4
Teacher_Model=Teacher()
#print(Teacher_Model)
Teacher_Model=Teacher_Model.to(device)
Teacher_optimizer = optim.Adam(Teacher_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc
import torchattacks
attack=torchattacks.CW(Teacher_Model,c=1, kappa=0, steps=10, lr=0.01)
def evaluate(model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    ADV_Dist=0
    all_preds = torch.tensor([])
    all_preds=all_preds.to(device)
    model.eval() # call model object for evaluation 
    #with torch.no_grad(): # Without computation of gredient 
    for (x,y) in iterator:
        x=x.float()
        x=x.to(device) # Transfer data to device 
        y=y.to(device) # Transfer label  to device 
        count=count+1
        adv_images=attack(x,y)
        adv_images=adv_images.to(device)
        L2_Dist=torch.norm(torch.abs(adv_images-x))
        x=x.detach()
        Predicted_Label = model(adv_images) # Predict claa label
        preds = (nn.functional.softmax(model(adv_images),dim=1)).max(1,keepdim=True)[1]
        all_preds = torch.cat((all_preds, preds.float()),dim=0) 
        loss = criterion(Predicted_Label, y) # Compute Loss 
        acc = calculate_accuracy(Predicted_Label, y) # compute Accuracy 
        #print("Validation Iteration Number=",count)
        epoch_loss += loss.item() # Compute Sum of  Loss 
        epoch_acc += acc.item() # Compute  Sum of Accuracy
        ADV_Dist=ADV_Dist+L2_Dist   
    return epoch_loss / len(iterator), epoch_acc / len(iterator),all_preds, ADV_Dist/len(iterator)
MODEL_SAVE_PATH = os.path.join("/home/mani/Desktop/AK/Empirical Study of Adversarial Examples for ESC/Urban8k/Trained Models", 'VGG16_CNN.pt')
Teacher_Model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # load the trained model 
def Class_Distribution(Class_Dist):
    arr=Class_Dist.detach().cpu().numpy()
    uniqueValues, occurCount = np.unique(arr, return_counts=True)
    occurCount=(occurCount/len(arr))*100
    print("Unique Classes=",uniqueValues)
    print("Class Distribution=",occurCount)
test_loss, test_acc,Class_Dist, ADV_Dist = evaluate(Teacher_Model, device, test_loader, criterion) # Compute Test Accuracy on Unseen Signals 
Class_Distribution(Class_Dist)
print("|Test Loss=",test_loss,"Test Accuracy=",test_acc*100)
print("L2 Norm Distance=",ADV_Dist)
