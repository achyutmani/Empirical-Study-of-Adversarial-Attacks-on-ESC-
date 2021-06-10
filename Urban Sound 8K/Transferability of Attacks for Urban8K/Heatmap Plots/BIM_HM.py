import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt
data = np.array([[100,99.03,91.33,98,97,100,98],[95,100,99,86,97,100,97.11],[95.19,98.07,100,97.11,97.11,100,99.03],[96.42,98.07,89.33,100,96,100,98],
[96,99.03,90.33,98,100,100,99],
	[96.84,100,88.33,98,98,100,99],[96,98,89.33,98,97,100,100]])
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
plt.savefig('BIM_HM_Us8K.pdf', bbox_inches='tight')
plt.show()
