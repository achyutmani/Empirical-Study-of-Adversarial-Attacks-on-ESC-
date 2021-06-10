import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt
data = np.array([[100,75,42,56,52,80,67],[63,100,44,54,49,76,68],[64,79,100,54,44,75,64],[63,74.33,42,100,41,80.33,67],[63,79.33,43,55,100,77.33,100],
	[63,78.33,43,57.99,45,100,64],[64,80.33,43,57.99,46,76.33,100]])
Temp=np.ones((7,7))*100	
data=Temp-data
data[np.diag_indices_from(data)]=100	
heat_map = sb.heatmap(data,annot=True, fmt=".2f",vmin=0, vmax=100,annot_kws={"size": 30})
Size=30
heat_map.set_xticklabels(['M1','M2','M3','M4','M5','M6','M7'],size=Size)
heat_map.set_yticklabels(['M1','M2','M3','M4','M5','M6','M7'],size=Size)
plt.ylabel('Source Model',size=Size)
plt.xlabel('Target Model',size=Size)
figure = plt.gcf()  # get current figure
figure.set_size_inches(20,10) # set figure's size manually to your full screen (32x18)
plt.savefig('FGSM_HM_Us8K.pdf', bbox_inches='tight')
plt.show()
