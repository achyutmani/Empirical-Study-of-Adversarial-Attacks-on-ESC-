import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt
data = np.array([[100,74.33,44,48,55,86.33,70],[63,100,43,46,53,84.34,70],[63,76.33,100,47,52,88.33,71],[63,78.33,43,100,53,87.33,71],
[63,77.33,43,47,100,87.33,70],
	[63,77.33,44,47,53,100,70],[63,75.64,43,48,54,88.33,100]])
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
plt.savefig('PGDR_HM_Us8K.pdf', bbox_inches='tight')
plt.show()
