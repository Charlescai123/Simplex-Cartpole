import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


modelpdv1 = np.transpose(np.loadtxt("safe_trajectory1.txt", skiprows=1, dtype=np.float32))
# modelpdvp5 = np.transpose(np.loadtxt("PDvp5.txt", delimiter=',', skiprows=1, dtype=np.float32))
# modelpdvn14 = np.transpose(np.loadtxt("PDvn14.txt", delimiter=',', skiprows=1, dtype=np.float32))
# modelpdvpn4 = np.transpose(np.loadtxt("PDvpn4.txt", delimiter=',', skiprows=1, dtype=np.float32))
#
# modelphydrlv1 = np.transpose(np.loadtxt("PhyDRLv1.txt", delimiter=',', skiprows=1, dtype=np.float32))
# modelphydrlvp5 = np.transpose(np.loadtxt("PhyDRLvp5.txt", delimiter=',', skiprows=1, dtype=np.float32))
# modelphydrlvn14 = np.transpose(np.loadtxt("PhyDRLvn14.txt", delimiter=',', skiprows=1, dtype=np.float32))
# modelphydrlvpn4 = np.transpose(np.loadtxt("PhyDRLvpn4.txt", delimiter=',', skiprows=1, dtype=np.float32))
#
# modeldrlv1 = np.transpose(np.loadtxt("DRLv1.txt", delimiter=',', skiprows=1, dtype=np.float32))
# modeldrlvp5 = np.transpose(np.loadtxt("DRLvp5.txt", delimiter=',', skiprows=1, dtype=np.float32))
# modeldrlvn14 = np.transpose(np.loadtxt("DRLvn14.txt", delimiter=',', skiprows=1, dtype=np.float32))
# modeldrlvpn4 = np.transpose(np.loadtxt("DRLvn4.txt", delimiter=',', skiprows=1, dtype=np.float32))



fig = plt.figure(figsize= (11, 8))

HR = 10000
he = np.arange(0.11, 0.37, 0.001)
sh1 = 0.17*np.ones(len(he))
sh2 = -0.17*np.ones(len(he))

ya = np.arange(-0.17, 0.17, 0.001)
ya1 = 0.37*np.ones(len(ya))
ya2 = 0.11*np.ones(len(ya))

plt.plot(x,y, linewidth=4, color='darkred')
plt.plot(he,sh1, linewidth=4, color='red')
plt.plot(he,sh2, linewidth=4, color='red')

plt.plot(ya1,ya, linewidth=4, color='red')
plt.plot(ya2,ya, linewidth=4, color='red')

plt.plot(modellmivp5[2][0:HR],modellmivp5[5][0:HR], linewidth=2, color='m',label = "Linear")
plt.plot(modelpdvp5[2][0:HR],modelpdvp5[5][0:HR], linewidth=2, color='y',label = "PD")
plt.plot(modeldrlvp5[2][0:HR],modeldrlvp5[5][0:HR], linewidth=2, color='limegreen',label = "DRL")
plt.plot(modelphydrlvp5[2][0:HR],modelphydrlvp5[5][0:HR], linewidth=2, color='blue',label = "Phy-DRL")

plt.legend(ncol=2)
plt.rc('legend',fontsize=25)

plt.xlabel("Height", fontsize=25)
plt.ylabel("Yaw", fontsize=25)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title("(b) Velocity 0.5 m/s, Snow Road", fontsize=25)
plt.grid()

plt.arrow(-0.16,0.0,0.25,0.0,width=.005)
plt.annotate('Safety Set',xy=(-0.02,0.025),horizontalalignment='center',fontsize=25)

plt.arrow(0.25,-0.401,-0.0,0.3,width=.005)
plt.annotate('Safety Envelope',xy=(0.24,-0.46),horizontalalignment='center',fontsize=25)

plt.arrow(modellmivp5[2][3835],modellmivp5[5][3835],modellmivp5[2][3836]-modellmivp5[2][3835],modellmivp5[5][3836]-modellmivp5[5][3835],width=.010,ec ='m',facecolor='m')
plt.arrow(modelpdvp5[2][1715],modelpdvp5[5][1715],modelpdvp5[2][1716]-modelpdvp5[2][1715],modelpdvp5[5][1716]-modelpdvp5[5][1715],width=.010,ec ='y',facecolor='y')
plt.arrow(modeldrlvp5[2][488],modeldrlvp5[5][488],modeldrlvp5[2][489]-modeldrlvp5[2][488],modeldrlvp5[5][489]-modeldrlvp5[5][488],width=.010,ec ='limegreen',facecolor='limegreen')
fig.savefig('ph2.pdf', dpi=600)






