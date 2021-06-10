import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt
data = np.array([[100,74,43,53,50,80,68],[63,100,42,50,44,73,68],[60.57,71.15,100,46.15,47.11,82.69,70.19],
[60.57,76.60,43,100,52,87.33,71],[36,75,42,49,100,85.33,68],
	[63,76.33,44,49,54,100,67],[64,75,43,49,52,86,100]])
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
plt.savefig('PGD_HM_Us8K.pdf', bbox_inches='tight')
plt.show()
