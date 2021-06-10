import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style 
style.use('seaborn-poster') #sets the size of the charts
style.use('seaborn-whitegrid')
#plt.style.use('seaborn')  
N = 4
ind = np.arange(N) 
width = 0.10
  
FGSM = [10,60,30,0]
bar1 = plt.bar(ind, FGSM, width, color = 'red',edgecolor = "black")
  
BIM = [30,40,30,0]
bar2 = plt.bar(ind+width, BIM, width, color='blue',edgecolor = "black")
  
PGD = [10,80,0,10]
bar3 = plt.bar(ind+width*2, PGD, width, color = 'magenta',edgecolor = "black")
PGDR = [10,80,0,0]
bar4 = plt.bar(ind+width*3, PGDR, width, color = 'black',edgecolor = "black")
plt.ylim(0,100) 
plt.xlabel("Class Labels",size=35)
plt.ylabel('Class Distriution(%)',size=35)
plt.rc('legend',fontsize=30)
plt.xticks(ind+width,['Air Conditioner', 'Drilling', 'Street Music','Siren'],size=35,rotation=10)
plt.yticks(size=35)
plt.legend( (bar1,bar2,bar3,bar4), ('FGSM', 'BIM', 'PGD','PGD-r'),loc="upper left" )
figure = plt.gcf()  # get current figure
figure.set_size_inches(20,10) # set figure's size manually to your full screen (32x18)
plt.savefig('US_All_M1.pdf', bbox_inches='tight')
plt.show()
