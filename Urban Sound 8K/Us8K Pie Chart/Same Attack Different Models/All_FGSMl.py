import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style 
style.use('seaborn-poster') #sets the size of the charts
style.use('seaborn-whitegrid')
#plt.style.use('seaborn')  
N = 8
ind = np.arange(N) 
width = 0.10
M1=[10,0,0,0,60,0,0,30]
M2=[10,0,20,0,30,0,0,30]
M3=[20,0,20,0,30,0,0,20]
M4=[20,10,20,10,20,0,0,20]
M5=[0,0,0,0,20,0,0,70]
M6=[10,0,0,10,20,10,10,20]
M7=[10,0,0,10,40,0,10,20]  
bar1 = plt.bar(ind, M1, width, color = 'tomato',edgecolor = "black")
bar2 = plt.bar(ind+width, M2, width, color='magenta',edgecolor = "black")
bar3 = plt.bar(ind+width*2, M3, width, color='springgreen',edgecolor = "black")
bar4 = plt.bar(ind+width*3, M4, width, color='green',edgecolor = "black")
bar5 = plt.bar(ind+width*4, M5, width, color='pink',edgecolor = "black")
bar6 = plt.bar(ind+width*5, M6, width, color='black',edgecolor = "black")
bar7 = plt.bar(ind+width*6, M7, width, color='blue',edgecolor = "black")
plt.ylim(0,100) 
plt.xlabel("Class Labels",size=35)
plt.ylabel('Class Distriution(%)',size=35)
plt.rc('legend',fontsize=30)
plt.xticks(ind+width,['Air Conditioner','Car Horn','Children Playing','Dog Bark','Drilling','Engine Idling','Siren','Street Music'],size=30,rotation=15)
plt.yticks(size=35)
plt.legend( (bar1,bar2,bar3,bar4,bar5,bar6,bar7), ('M1','M2','M3','M4','M5','M6','M7'),loc="upper left" )
figure = plt.gcf()  # get current figure
figure.set_size_inches(20,10) # set figure's size manually to your full screen (32x18)
plt.savefig('US_FGSM_All.pdf', bbox_inches='tight')
plt.show()
