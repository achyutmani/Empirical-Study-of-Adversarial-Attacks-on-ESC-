import os
import torch
import numpy as np 
import h5py
from torch.utils.data import dataset
import pandas as pd 
from PIL import Image
class LAEData_Train():
	def __init__(self,transform=None):
		self.annotations=np.array(pd.read_csv("/home/mani/Desktop/AK/Empirical Study of Adversarial Examples for ESC/Urban8k/Urban Metadata/Train.csv"))# Read The names of Training Signals
		self.transform=transform
	def __len__(self):
		return len(self.annotations)
	def __getitem__(self,index):
		key=self.annotations[index,0]
		#print(index)
		with h5py.File('/home/mani/Desktop/AK/Empirical Study of Adversarial Examples for ESC/Urban8k/Urban Metadata/Urban.hdf5', 'r') as f: # Database for Left Channel Training Spectrogram
			SG_Data = f[key][()]
			#SG_Data=np.array(SG_Data)
			SG_Data=Image.fromarray(SG_Data)
			#SG_Data=np.log((SG_Data + 1e-10))
			#for i in range(len(SG_Data)):
				#SG_Data[i,:]=SG_Data[i,:]/np.sum(SG_Data[i,:])
			#SGmax,SGmin=SG_Data.max(),SG_Data.min()
			#SG_Data=(SG_Data-SGmin)/(SGmax-SGmin)
			SG_Label= torch.from_numpy(np.array(self.annotations[index,1]))
			if self.transform:
				SG_Data=self.transform(SG_Data)
		return (SG_Data,SG_Label)	

		
		
