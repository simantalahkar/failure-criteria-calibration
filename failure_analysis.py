from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#from numpy import polyfit
from matplotlib import cm

from mpl_toolkits import mplot3d


mpl.rcParams['mathtext.fontset'] = 'stixsans'
mpl.rcParams['mathtext.default'] = 'it'
mpl.rcParams['font.family'] = 'sans serif'
mpl.rcParams['font.size'] = 8
mpl.rcParams['legend.fontsize'] = 'medium'		#relative size to font.size
mpl.rcParams['figure.titlesize'] = 'large'     #relative size to font.size

##################################################################################################
############################## Read multiaxial failure stress data and solve for coefficients of Tsai-Wu criterion + calculate the values of optimized TW function for rest of failure data points

all_data=np.loadtxt('failure_points.dat', delimiter=' ', comments='#', skiprows=0, max_rows=None)
calc_set=np.loadtxt('best_set.dat', delimiter=' ', comments='#', skiprows=0, max_rows=None)

strengths = np.array([[a[0]**2, a[0], 2*a[0]*a[1], a[1]**2, a[1], 2*a[1]*a[2], a[2]**2, a[2], 2*a[0]*a[2]] for a in calc_set])

tx, ty, tz = all_data[:,0], all_data[:,1], all_data[:,2] 
xstrengths0, ystrengths0, zstrengths0 = calc_set[:,0], calc_set[:,1], calc_set[:,2] 


#print(strengths)
b=np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1]])
#print(b)
sinv=np.linalg.inv(strengths)
#print(sinv)
coefficients=np.dot(sinv,b)
print(coefficients)
#print(np.dot(sinv,strengths))

F11=coefficients[0]
F1=coefficients[1]
F12=coefficients[2]
F22=coefficients[3]
F2=coefficients[4]
F23=coefficients[5]
F33=coefficients[6]
F3=coefficients[7]
F13=coefficients[8]

D12=F12**2-F11*F22
D23=F23**2-F33*F22
D13=F13**2-F11*F33
print('D12={}'.format(D12))
print('D23={}'.format(D23))
print('D13={}'.format(D13))

pointx=[]
pointy=[]
pointz=[]

print('Number of all stress points={}'.format(len(tx)))

for i in range(len(tx)):
	flag=0
#	print('i={}'.format(i))
	for j in range(len(xstrengths0)):
		if ((xstrengths0[j]==tx[i]) and (ystrengths0[j]==ty[i]) and (zstrengths0[j]==tz[i])):
			flag=1
#			print(flag)
			
	if flag==0:
		pointx.append(tx[i])
		pointy.append(ty[i])
		pointz.append(tz[i])
pointx=np.array(pointx)
pointy=np.array(pointy)
pointz=np.array(pointz)

print('Number of remaining stress points to test={}'.format(len(pointx)))

TWtest=np.array((F11*pointx**2 + F1*pointx + F22*pointy**2 \
	+ F2*pointy + F33*pointz**2 + F3*pointz + 2*F12*pointx*pointy \
	 + 2*F23*pointz*pointy + 2*F13*pointx*pointz))

########################################################################
###############################################################################################
#####################################################################################
##############################strengths


s1t=0.5*(-F1+np.sqrt(F1**2+4*F11))/F11
s1c=1/(s1t*F11)

s2t=0.5*(-F2+np.sqrt(F2**2+4*F22))/F22
s2c=1/(s2t*F22)

s3t=0.5*(-F3+np.sqrt(F3**2+4*F33))/F33
s3c=1/(s3t*F33)

print('Magnitudes of s1t={}, s1c={}, s2t={}, s2c={}, s3t={}, s3c={}'.format(s1t,s1c,s2t,s2c,s3t,s3c))
#######################################################
Y1=s1t
Y2=s2t
Y3=s3c

A=0.5*(1/Y2**2 + 1/Y3**2 - 1/Y1**2)
B=0.5*(1/Y3**2 + 1/Y1**2 - 1/Y2**2)
C=0.5*(1/Y1**2 + 1/Y2**2 - 1/Y3**2)
#######################################################
Dp=-4*(A*B+A*C+B*C)
m11=2*B+2*C
m12=-2*C
m13=-2*B
m22=2*A+2*C
m23=-2*A
m33=2*A+2*B
#print(A)
matrix=np.array([[m11[0],m12[0],m13[0]],[m12[0],m22[0],m23[0]],[m13[0],m23[0],m33[0]]])
#print(matrix)

detmatrixa=np.linalg.det(matrix)
#print('determinant of hills min matrix domain 1 : {}'.format(detmatrixa))

print('Magnitudes of A={}, B={}, C={}, Dp={} for domain 1'.format(A,B,C,Dp))
#print('Magnitudes of A+B={}, B+C={}, C+A={} for domain 1'.format((B+A),(B+C),(C+A)))


###############################################################################################
###########################################################################################
############## shear
################## for 12 ###################

sza=-102.1
sxa=-34.3
sya=34.4
sxya=63.6

F66=(1-F33*sza**2-F3*sza-F11*sxa**2-F1*sxa-F22*sya**2-F2*sya-2*F12*sxa*sya \
	-2*F13*sxa*sza-2*F23*sza*sya)/(sxya**2)
s12=1/np.sqrt(F66)
print('\n F66= {}, sxy= {}'.format(F66,s12))

#print(-(-F33*sza**2-F3*sza-F11*sxa**2-F1*sxa-F22*sya**2-F2*sya-2*F12*sxa*sya \
#	-2*F13*sxa*sza-2*F23*sza*sya))


F=(1-A*(sya-sza)**2-B*(sza-sxa)**2-C*(sxa-sya)**2)/(2*sxya**2)
s12hills=1/np.sqrt(2*F)
print('F= {}, sxyhills= {}'.format(F,s12hills))

################## for 23 ###################
#szb=1.9
#sxb=15.1
#syb=106.04
#syzb=39.8

szb=-102.7
sxb=0
syb=102.8
syzb=-74.01

F44=(1-F33*szb**2-F3*szb-F11*sxb**2-F1*sxb-F22*syb**2-F2*syb-2*F12*sxb*syb \
	-2*F13*sxb*szb-2*F23*szb*syb)/(syzb**2)
	
#print(-(-F33*szb**2-F3*szb-F11*sxb**2-F1*sxb-F22*syb**2-F2*syb-2*F12*sxb*syb \
#	-2*F13*sxb*szb-2*F23*szb*syb))

#s23=1/np.sqrt(F44)
#print('F44= {}, syz= {}'.format(F44,s23))

#D=(1-A*(syb-szb)**2-B*(szb-sxb)**2-C*(sxb-syb)**2)/(2*syzb**2)
#s23hills=1/np.sqrt(2*D)
#print('D= {}, syzhills= {}'.format(D,s23hills))

################## for 13 ###################
szc=-87.3
sxc=87.3
syc=0
sxzc=66.1

#szc=0.5
#sxc=93.5
#syc=12.4
#sxzc=29.7

F55=(1-F33*szc**2-F3*szc-F11*sxc**2-F1*sxc-F22*syc**2-F2*syc-2*F12*sxc*syc \
	-2*F13*sxc*szc-2*F23*szc*syc)/(sxzc**2)

s13=1/np.sqrt(F55)
print('F55= {}, sxz= {} \n'.format(F55,s13))
#print(-(-F33*szc**2-F3*szc-F11*sxc**2-F1*sxc-F22*syc**2-F2*syc-2*F12*sxc*syc \
#	-2*F13*sxc*szc-2*F23*szc*syc))


#E=(1-A*(syc-szc)**2-B*(szc-sxc)**2-C*(sxc-syc)**2)/(2*sxzc**2)
#s13hills=1/np.sqrt(2*E)
#print('E= {}, sxzhills= {}'.format(E,s13hills))

###############################################################################################
##################################################################################### 
########################### Hills 


hills=[]
for i in range(len(pointx)):

	hills.append(A*(pointy[i]-pointz[i])**2+B*(pointx[i]-pointz[i])**2+C*(pointy[i]-pointx[i])**2)
	
hills=np.array(hills)


############################################################################################################################
############################################################################################################################
################### minimum value of tsai wu 
twdoubles=np.array([[F11[0],F12[0],F13[0]],[F12[0],F22[0],F23[0]],[F13[0],F23[0],F33[0]]])
#print(twdoubles)
twsingles=np.array([[-0.5*F1[0]],[-0.5*F2[0]],[-0.5*F3[0]]])
#print(twsingles)
#print(np.linalg.inv(twdoubles))
#print(np.linalg.det(twdoubles))
twminpoint=np.dot(np.linalg.inv(twdoubles),twsingles)
#print('tw envelope mid point')
#print(twminpoint)

twmin=F11*twminpoint[0]**2 + F1*twminpoint[0] + F22*twminpoint[1]**2 \
	+ F2*twminpoint[1] + F33*twminpoint[2]**2 + F3*twminpoint[2] + 2*F12*twminpoint[0]*twminpoint[1] \
	 + 2*F23*twminpoint[2]*twminpoint[1] + 2*F13*twminpoint[0]*twminpoint[2]

print('minimum of tsai wu={}'.format(twmin))

Futw=1-twmin
print('length of tsai wu value until 1')
print(Futw)

Amat=np.array([[F11[0],F12[0],F13[0],F1[0]/2],[F12[0],F22[0],F23[0],F2[0]/2],[F13[0],F23[0],F33[0],F3[0]/2],[F1[0]/2,F2[0]/2,F3[0]/2,-1.0]])
values , vectors = np.linalg.eig(twdoubles)
print('eigen values, of characteristic form matrix: \n', values)
print('eigen vectors of characteristic form matrix: \n', vectors)

vc=np.transpose(vectors)
total=np.append([values[:]],vc[:],axis=0)
total=total[:,total[0,:].argsort()]
vsorted=np.flip(total,axis=1)

print('total sorted matrix: \n', vsorted)

Ddet=np.linalg.det(twdoubles)
Adet=np.linalg.det(Amat)

x0mat=np.array([[F1[0]/2,F12[0],F13[0]],[F2[0]/2,F22[0],F23[0]],[F3[0]/2,F23[0],F33[0]]])
y0mat=np.array([[F11[0],F1[0]/2,F13[0]],[F12[0],F2[0]/2,F23[0]],[F13[0],F3[0]/2,F33[0]]])
z0mat=np.array([[F11[0],F12[0],F1[0]/2],[F12[0],F22[0],F2[0]/2],[F13[0],F23[0],F3[0]/2]])
x0=-np.linalg.det(x0mat)/Ddet
y0=-np.linalg.det(y0mat)/Ddet
z0=-np.linalg.det(z0mat)/Ddet

print('Coordinates of TW center recalculated: ({},{},{})'.format(x0,y0,z0))

a=np.sqrt(-Adet/(Ddet*vsorted[0][2]))
b=np.sqrt(-Adet/(Ddet*vsorted[0][1]))
c=np.sqrt(-Adet/(Ddet*vsorted[0][0]))

print('value of A/D = {}'.format(Adet/Ddet))

print('Principle axis/diameters full lengths: 2a = {}, 2b = {}, 2c = {}'.format(2*a,2*b,2*c))


############################################################################################################################
############################################################################################################################
################### minimum value of hills

hillsdoubles=np.array([[B[0]+C[0],-C[0],-B[0]],[-C[0],A[0]+C[0],-A[0]],[-B[0],-A[0],A[0]+B[0]]])

Amathills=np.array([[B[0]+C[0],-C[0],-B[0],0],[-C[0],A[0]+C[0],-A[0],0],[-B[0],-A[0],A[0]+B[0],0],[0,0,0,-1.0]])
valueshills , vectorshills = np.linalg.eig(hillsdoubles)
print('eigen valuesof hills matrix: \n', valueshills)
print('eigen vectors of hills matrix: \n', vectorshills)

vchills=np.transpose(vectorshills)
totalhills=np.append([valueshills[:]],vchills[:],axis=0)
totalhills=totalhills[:,totalhills[0,:].argsort()]
vsortedhills=np.flip(totalhills,axis=1)

print('total hills sorted matrix: \n', vsortedhills)

Ddethills=np.linalg.det(hillsdoubles)
Adethills=np.linalg.det(Amathills)

x0mathills=np.array([[0,-C[0],-B[0]],[0,A[0]+C[0],-A[0]],[0,-A[0],A[0]+B[0]]])
y0mathills=np.array([[B[0]+C[0],0,-B[0]],[-C[0],0,-A[0]],[-B[0],0,A[0]+B[0]]])
z0mathills=np.array([[B[0]+C[0],-C[0],0],[-C[0],A[0]+C[0],0],[-B[0],-A[0],0]])
x0hills=-np.linalg.det(x0mathills)/Ddethills
y0hills=-np.linalg.det(y0mathills)/Ddethills
z0hills=-np.linalg.det(z0mathills)/Ddethills

#print('Coordinates of hills center recalculated: ({},{},{})'.format(x0hills,y0hills,z0hills))

ahills=np.sqrt(-Adethills/(Ddethills*vsortedhills[0][2]))
bhills=np.sqrt(-Adethills/(Ddethills*vsortedhills[0][1]))
chills=np.sqrt(-Adethills/(Ddethills*vsortedhills[0][0]))

print('value of hills A/D = {}'.format(Adethills/Ddethills))

print('Principle axis/diameters full lengths hills: 2a = {}, 2b = {}, 2c = {}'.format(2*ahills,2*bhills,2*chills))



############################################################################################################################
############################################################################################################################
###################### distributions


arrone=np.ones(len(TWtest))
dTWtest=TWtest-1
dhills=hills-1

	 
index=np.linspace(0,1,len(TWtest))


total_percenthills=100*np.sqrt(dhills+1)			#since minimum of hills is 0 so max value difference till 1 is 1
#print(total_percenthills)
#print(index)
intervalTWtest=np.sqrt(dTWtest/Futw[0]+1)-1

total_percentTWtest=100*(1+intervalTWtest)

relproxTW=total_percentTWtest/100

relproxhills=total_percenthills/100

deviationTWtest=intervalTWtest*100

deviationhills=100*(np.sqrt(dhills+1)-1)


j=0
for i in total_percentTWtest:
	if i>100:
		total_percentTWtest[j]=200-i
	j=j+1

TW98acyvalue=(0.98**2-1)*Futw[0]+1
print('\n TW 98% accuracy point={}'.format(TW98acyvalue))

hills98acyvalue=0.98**2
print('hills 98% accuracy point={}'.format(hills98acyvalue))



displacement=np.sqrt((pointx-x0)**2+(pointy-y0)**2+(pointz-z0)**2)
displacementhills=np.sqrt((pointx)**2+(pointy)**2+(pointz)**2)

stressmagTW=np.sqrt((pointx)**2+(pointy)**2+(pointz)**2)

TWcorrection=[]
hillscorrection=[]

displace_TWdev=[]
j=0
for i in deviationTWtest:
	displace_TWdev.append(0.01*i*displacement[j]/(1+0.01*i))
	TWcorrection.append(-displace_TWdev[-1]/stressmagTW[j])
	j=j+1
	
displace_hillsdev=[]
j=0
for i in deviationhills:
	displace_hillsdev.append(0.01*i*displacementhills[j]/(1+0.01*i))
	hillscorrection.append(-displace_hillsdev[-1]/displacementhills[j])
	j=j+1



mxpercent=[]
mxdev=[]
mxcorrection=[]
for i in range(len(pointx)):
	if pointx[i]<0:
		s1=-s1c
	else:
		s1=s1t
	if pointy[i]<0:
		s2=-s2c
	else:
		s2=s2t
	if pointz[i]<0:
		s3=-s3c
	else:
		s3=s3t		
		
	mxpercent.append(100*max((pointx[i]/s1),(pointy[i]/s2),(pointz[i]/s3)))
	
	if (((pointx[i]/s1)>=(pointy[i]/s2)) and ((pointx[i]/s1)>=(pointz[i]/s3))):
		mxdev.append(np.abs(pointx[i])-np.abs(s1))
		mxcorrection.append(-mxdev[-1]/np.abs(pointx[i]))
	elif (((pointy[i]/s2)>=(pointx[i]/s1)) and ((pointy[i]/s2)>=(pointz[i]/s3))):
		mxdev.append(np.abs(pointy[i])-np.abs(s2))
		mxcorrection.append(-mxdev[-1]/np.abs(pointy[i]))
	elif (((pointz[i]/s3)>=(pointy[i]/s2)) and ((pointz[i]/s3)>=(pointx[i]/s1))):
		mxdev.append(np.abs(pointz[i])-np.abs(s3))
		mxcorrection.append(-mxdev[-1]/np.abs(pointz[i]))


meanTWtest_acy=np.mean(total_percentTWtest)
stdTWtest_acy=np.std(total_percentTWtest)

meanTWtest_devn=np.mean(deviationTWtest)
stdTWtest_devn=np.std(deviationTWtest)

meanhills_devn=np.mean(deviationhills)
stdhills_devn=np.std(deviationhills)

meanTWtest=np.mean(TWtest)
stdTWtest=np.std(TWtest)

meanhills_acy=np.mean(total_percenthills)
stdhills_acy=np.std(total_percenthills)


meanTWdev=np.mean(displace_TWdev)
stdTWdev=np.std(displace_TWdev)

meanhillsdev=np.mean(displace_hillsdev)
stdhillsdev=np.std(displace_hillsdev)

meanhills=np.mean(hills)
stdhills=np.std(hills)
mxcorrection=np.array(mxcorrection)
TWcorrection=np.array(TWcorrection)
hillscorrection=np.array(hillscorrection)
mxpercent=np.array(mxpercent)
mxdev=np.array(mxdev)

meanmxpercent=np.mean(mxpercent)
stdmxpercent=np.std(mxpercent)

stressmagTW=np.array(stressmagTW)

meanstressmag=np.mean(stressmagTW)
stdstressmag=np.std(stressmagTW)

meanmxdev=np.mean(mxdev)
stdmxdev=np.std(mxdev)

meanmxcorrection=np.mean(-100*mxcorrection)
stdmxcorrection=np.std(-100*mxcorrection)

meanTWcorrection=np.mean(-100*TWcorrection)
stdTWcorrection=np.std(-100*TWcorrection)
meanhillscorrection=np.mean(-100*hillscorrection)
stdhillscorrection=np.std(-100*hillscorrection)

meanrelproxTW=np.mean(relproxTW)
stdrelproxTW=np.std(relproxTW)
meanrelproxhills=np.mean(relproxhills)
stdrelproxhills=np.std(relproxhills)



print('\n \n Max percent mean ={} +/- {} %'.format(meanmxpercent,stdmxpercent))
print('max stress deviations mean ={} +/- {} GPa \n \n'.format(meanmxdev,stdmxdev))

print('\n \n TWtest deviations mean ={} +/- {} GPa'.format(meanTWdev,stdTWdev))
print('Hills deviations mean ={} +/- {} GPa \n \n'.format(meanhillsdev,stdhillsdev))

print('\n \n Corrections for max stress mean ={} +/- {}%'.format(meanmxcorrection,stdmxcorrection))
print('\n \n Corrections for TW mean ={} +/- {}%'.format(meanTWcorrection,stdTWcorrection))
print('\n \n Corrections for hills mean ={} +/- {}%'.format(meanhillscorrection,stdhillscorrection))
print('\n \n Magnitude of stresses mean ={} +/- {} GPa'.format(meanstressmag,stdstressmag))
#print('Hills deviations mean ={} +/- {} GPa \n \n'.format(meanhillsdev,stdhillsdev))

print('TWtest mean relative proximity={}, std={}'.format(meanrelproxTW,stdrelproxTW))
print('Hills mean relative proximity={} +/- {} \n'.format(meanrelproxhills,stdrelproxhills))
print('TWtest mean ={}, std={}'.format(meanTWtest,stdTWtest))
print('hills mean ={}, std={}'.format(meanhills,stdhills))


print('Hills mean accuracy={}, std={} %'.format(meanhills_acy,stdhills_acy))
print('TWtest mean accuracy={}, std={} %'.format(meanTWtest_acy,stdTWtest_acy))
####################################################################################################
######################################################### distribution
fig = plt.figure(figsize=(21,13))
ax1=fig.add_subplot(431, aspect='auto', adjustable='datalim')
ax1.scatter(index,TWtest,c="blue", s=10 , label="F", marker="s")
ax1.set_ylabel("Function value")
ax1.set_ylim(0,6.5)
plt.xticks([])
plt.legend(loc="upper left")
ax2=fig.add_subplot(432, aspect='auto', adjustable='datalim')
ax2.scatter(index,hills,c="green", s=10 , label="Hill", marker="s")
ax2.set_ylim(0,6.5)
plt.xticks([])
plt.yticks([])
plt.legend(loc="upper left")
ax3=fig.add_subplot(433, aspect='auto', adjustable='datalim')
ax3.scatter(index,mxpercent/100,c="red", s=10 , label="Max stress", marker="s")
ax3.set_ylim(0,6.5)
plt.xticks([])
plt.yticks([])
plt.legend(loc="upper left")

ax4=fig.add_subplot(434, aspect='auto', adjustable='datalim')
ax4.scatter(index,relproxTW,c="blue", s=10 , label="F", marker="o")
ax4.set_ylabel(r"Relative proximity, $\gamma$")
ax4.set_ylim(0.48,2.9)
plt.xticks([])
plt.legend(loc="upper left")
ax5=fig.add_subplot(435, aspect='auto', adjustable='datalim')
ax5.scatter(index,relproxhills,c="green", s=10 , label="Hill", marker="o")
ax5.set_ylim(0.48,2.9)
plt.xticks([])
plt.yticks([])
plt.legend(loc="upper left")
ax6=fig.add_subplot(436, aspect='auto', adjustable='datalim')
ax6.scatter(index,mxpercent/100,c="red", s=10 , label="Max stress", marker="o")
ax6.set_ylim(0.48,2.9)
plt.xticks([])
plt.yticks([])
plt.legend(loc="upper left")

ax7=fig.add_subplot(437, aspect='auto', adjustable='datalim')
ax7.scatter(index,displace_TWdev,c="blue", s=10 , label="F", marker="^")
ax7.set_ylabel(r"Offset, $\Delta \Sigma$ (GPa)")
ax7.set_ylim(-102,300)
plt.xticks([])
plt.legend(loc="upper left")
ax8=fig.add_subplot(438, aspect='auto', adjustable='datalim')
ax8.scatter(index,displace_hillsdev,c="green", s=10 , label="Hill", marker="^")
ax8.set_ylim(-102,300)
plt.xticks([])
plt.yticks([])
plt.legend(loc="upper left")
ax9=fig.add_subplot(439, aspect='auto', adjustable='datalim')
ax9.scatter(index,mxdev,c="red", s=10 , label="Max stress", marker="^")
ax9.set_ylim(-102,300)
plt.xticks([])
plt.yticks([])
plt.legend(loc="upper left")

ax10=fig.add_subplot(4,3,10, aspect='auto', adjustable='datalim')
ax10.scatter(index,-100*TWcorrection,c="blue", s=12 , label="F", marker="*")
ax10.set_ylabel(r"Percent Offset, $d$ (%)")
ax10.set_ylim(-105,68)
plt.xticks([])
plt.legend(loc="upper left")
ax11=fig.add_subplot(4,3,11, aspect='auto', adjustable='datalim')
ax11.scatter(index,-100*hillscorrection,c="green", s=12 , label="Hill", marker="*")
ax11.set_ylim(-105,68)
plt.xticks([])
plt.yticks([])
plt.legend(loc="upper left")
ax12=fig.add_subplot(4,3,12, aspect='auto', adjustable='datalim')
ax12.scatter(index,-100*mxcorrection,c="red", s=12 , label="Max stress", marker="*")
ax12.set_ylim(-105,68)
plt.xticks([])
plt.yticks([])
plt.legend(loc="upper left")



plt.savefig('28_dist.png')
plt.show()


##############################################################################################################

##############################################################################################################

fig = plt.figure(figsize=(21.36,13.28))
ax1=fig.add_subplot(241, aspect='auto', adjustable='datalim')
ax1.scatter(index,100*mxcorrection)
ax1.set_title('% Corrections max stress', fontsize=10);

ax2=fig.add_subplot(242, aspect='auto', adjustable='datalim')
ax2.scatter(index,100*TWcorrection)
ax2.set_title('% Corrections TW', fontsize=10);

ax3=fig.add_subplot(243, aspect='auto', adjustable='datalim')
ax3.scatter(index,100*hillscorrection)
ax3.set_title('% Corrections hills', fontsize=10);

ax4=fig.add_subplot(244, aspect='auto', adjustable='datalim')
ax4.scatter(index,stressmagTW)
ax4.set_title('Magnitude of stresses', fontsize=10);

ax5=fig.add_subplot(245, aspect='auto', adjustable='datalim')
ax5.scatter(index,pointx)
ax5.set_title('x stress', fontsize=10);

ax6=fig.add_subplot(246, aspect='auto', adjustable='datalim')
ax6.scatter(index,pointy)
ax6.set_title('y stress', fontsize=10);

ax7=fig.add_subplot(247, aspect='auto', adjustable='datalim')
ax7.scatter(index,pointz)
ax7.set_title('z stress', fontsize=10);

ax8=fig.add_subplot(248, aspect='auto', adjustable='datalim')
ax8.scatter(index,displacement)
ax8.set_title('displacements from center', fontsize=10);


plt.savefig('28_stress_distribution.png')
#plt.show()

fig = plt.figure(figsize=(21.36,13.28))


ax3=fig.add_subplot(241, aspect='auto', adjustable='datalim')
ax3.scatter(index,deviationTWtest)
ax3.set_title('TW % proximities', fontsize=10);

ax4=fig.add_subplot(242, aspect='auto', adjustable='datalim')
ax4.scatter(index,TWtest)
ax4.set_title('TW scatter', fontsize=10);

ax1=fig.add_subplot(243, aspect='auto', adjustable='datalim')
ax1.scatter(index,displace_TWdev)
ax1.set_title('TW deviations', fontsize=10);

ax2=fig.add_subplot(247, aspect='auto', adjustable='datalim')
ax2.scatter(index,displace_hillsdev)
ax2.set_title('Hills deviations', fontsize=10);

ax5=fig.add_subplot(244, aspect='auto', adjustable='datalim')
ax5.scatter(index,mxpercent/100)
ax5.set_title('Max stress scatter', fontsize=10);

ax6=fig.add_subplot(248, aspect='auto', adjustable='datalim')
ax6.scatter(index,mxdev)
ax6.set_title('max percent deviations', fontsize=10);


ax7=fig.add_subplot(245, aspect='auto', adjustable='datalim')
ax7.scatter(index,deviationhills)
ax7.set_title('hills % proximities', fontsize=10);

ax8=fig.add_subplot(246, aspect='auto', adjustable='datalim')
ax8.scatter(index,hills)
ax8.set_title('hills scatter', fontsize=10);

plt.savefig('28_distribution.png')
#plt.show()



############################################################################################################################
############################################################################################################################

################################ surface plot

# x and y axis
x = np.linspace(-600,200, 36)
y = np.linspace(-600,200, 36)
z = np.linspace(-200, 50, 12)

# function for z axea
def fzp(x, y):
	a=F33
	b=F3+2*F13*x+2*F23*y
	c=F22*y**2+F2*y+F11*x**2+F1*x+2*F12*x*y-1
	det=b**2-4*a*c
	return (-b+np.sqrt(det))/(2*a)
def fzn(x, y):
	a=F33
	b=F3+2*F13*x+2*F23*y
	c=F22*y**2+F2*y+F11*x**2+F1*x+2*F12*x*y-1
	det=b**2-4*a*c
	return (-b-np.sqrt(det))/(2*a)

def fyp(x, z):
	a=F22
	b=F2+2*F12*x+2*F23*z
	c=F33*z**2+F3*z+F11*x**2+F1*x+2*F13*x*z-1
	det=b**2-4*a*c
	return (-b+np.sqrt(det))/(2*a)
def fyn(x, z):
	a=F22
	b=F2+2*F12*x+2*F23*z
	c=F33*z**2+F3*z+F11*x**2+F1*x+2*F13*x*z-1
	det=b**2-4*a*c
	return (-b-np.sqrt(det))/(2*a)

def fxp(y, z):
	a=F11
	b=F1+2*F12*y+2*F13*z
	c=F33*z**2+F3*z+F22*y**2+F2*y+2*F23*y*z-1
	det=b**2-4*a*c
	return (-b+np.sqrt(det))/(2*a)
def fxn(y, z):
	a=F11
	b=F1+2*F12*y+2*F13*z
	c=F33*z**2+F3*z+F22*y**2+F2*y+2*F23*y*z-1
	det=b**2-4*a*c
	return (-b-np.sqrt(det))/(2*a)



fig = plt.figure(figsize=(21.36,13.28))
ax1=fig.add_subplot(121, aspect='auto', adjustable='datalim',projection ='3d')
#ax = plt.axes(projection ='3d')
p = ax1.scatter(pointx, pointy, pointz, c = total_percentTWtest,  alpha=1)	#vmin = 0,


fig.colorbar(p, shrink=0.2, aspect=10)

ax1.set_xlabel(r'x stress, $\sigma_x$ (GPa)', fontsize=11)
ax1.set_ylabel(r'y stress, $\sigma_y$ (GPa)', fontsize=11)
ax1.set_zlabel(r'z stress, $\sigma_z$ (GPa)', fontsize=11)

ax1.set_zlim(-300,50)
ax1.set_xlim(-500,200)
ax1.set_ylim(-500,200)


xlinex=[-1000,500]
yliney=[-1000,500]
zlinez=[-1000,100]
#ax1.plot(xlinex,[0,0],[0,0])
#ax1.plot([0,0],yliney,[0,0])
#ax1.plot([0,0],[0,0],zlinez)
#ax.axhline(color='#d62728')
ax1.set_title('Failure points', fontsize=12);


X1, Y1 = np.meshgrid(x, y)
Z1p = fzp(X1, Y1)
Z1n = fzn(X1, Y1)

X2, Z2 = np.meshgrid(x, z)
Y2p = fyp(X2, Z2)
Y2n = fyn(X2, Z2)

Y3, Z3 = np.meshgrid(y, z)
X3p = fxp(Y3, Z3)
X3n = fxn(Y3, Z3)

#fig = plt.figure()
ax2=fig.add_subplot(122, aspect='auto', adjustable='datalim',projection ='3d')
ax2.plot_wireframe(X1, Y1, Z1p, color ='orange', alpha=0.15)
ax2.plot_wireframe(X1, Y1, Z1n, color ='orange', alpha=0.15)

ax2.plot_wireframe(X2, Y2p, Z2, color ='orange', alpha=0.15)
ax2.plot_wireframe(X2, Y2n, Z2, color ='orange', alpha=0.15)

ax2.plot_wireframe(X3p, Y3, Z3, color ='orange', alpha=0.15)
ax2.plot_wireframe(X3n, Y3, Z3, color ='orange', alpha=0.15)


ax2.scatter(pointx, pointy, pointz, color = 'r',alpha=0.5)
ax2.scatter(xstrengths0, ystrengths0,zstrengths0, color='b', alpha=1)#, color ='orange')


#ax2.plot(xlinex,[0,0],[0,0])
#ax2.plot([0,0],yliney,[0,0])
#ax2.plot([0,0],[0,0],zlinez)
ax2.set_zlim(-300,50)
ax2.set_xlim(-500,200)
ax2.set_ylim(-500,200)

ax2.set_xlabel(r'x stress, $\sigma_x$ (GPa)', fontsize=11)
ax2.set_ylabel(r'y stress, $\sigma_y$ (GPa)', fontsize=11)
ax2.set_zlabel(r'z stress, $\sigma_z$ (GPa)', fontsize=11)


ax2.set_title('Quadratic failure Envelope', fontsize=12);
plt.savefig('26_3D.png')
plt.show()
plt.close()

#plt.plot(index,TWtest)
#plt.show()
#plt.close()


############################################################################################################################
############################################################################################################################

################################ surface plot HILLS

# x and y axis
x = np.linspace(-600,600, 52)
y = np.linspace(-600,600, 52)
z = np.linspace(-200, 200, 20)

# function for z axea
def fzp(x, y):
	a=A+B
	b=-2*(A*y+B*x)
	c=(A+C)*y**2+(B+C)*x**2-2*C*x*y-1
	det=b**2-4*a*c
	return (-b+np.sqrt(det))/(2*a)
def fzn(x, y):
	a=A+B
	b=-2*(A*y+B*x)
	c=(A+C)*y**2+(B+C)*x**2-2*C*x*y-1
	det=b**2-4*a*c
	return (-b-np.sqrt(det))/(2*a)

def fyp(x, z):
	a=A+C
	b=-2*(A*z+C*x)
	c=(A+B)*z**2+(B+C)*x**2-2*B*z*x-1
	det=b**2-4*a*c
	return (-b+np.sqrt(det))/(2*a)
def fyn(x, z):
	a=A+C
	b=-2*(A*z+C*x)
	c=(A+B)*z**2+(B+C)*x**2-2*B*z*x-1
	det=b**2-4*a*c
	return (-b-np.sqrt(det))/(2*a)

def fxp(y, z):
	a=B+C
	b=-2*(B*z+C*y)
	c=(A+B)*z**2+(A+C)*y**2-2*A*y*z-1
	det=b**2-4*a*c
	return (-b+np.sqrt(det))/(2*a)
def fxn(y, z):
	a=B+C
	b=-2*(B*z+C*y)
	c=(A+B)*z**2+(A+C)*y**2-2*A*y*z-1
	det=b**2-4*a*c
	return (-b-np.sqrt(det))/(2*a)



fig = plt.figure(figsize=(21.36,13.28))
ax1=fig.add_subplot(121, aspect='auto', adjustable='datalim',projection ='3d')
#ax = plt.axes(projection ='3d')
p = ax1.scatter(pointx, pointy, pointz, c = total_percenthills,  alpha=1)	#vmin = 0,


fig.colorbar(p, shrink=0.2, aspect=10)

ax1.set_xlabel(r'x failure stress, $\sigma_x$ (GPa)', fontsize=11)
ax1.set_ylabel(r'y failure  stress, $\sigma_y$ (GPa)', fontsize=11)
ax1.set_zlabel(r'z failure  stress, $\sigma_z$ (GPa)', fontsize=11)

ax1.set_zlim(-200,200)
ax1.set_xlim(-600,600)
ax1.set_ylim(-600,600)


xlinex=[-1000,500]
yliney=[-1000,500]
zlinez=[-1000,100]
ax1.plot(xlinex,[0,0],[0,0])
ax1.plot([0,0],yliney,[0,0])
ax1.plot([0,0],[0,0],zlinez)
#ax.axhline(color='#d62728')
ax1.set_title('Failure points', fontsize=12);


X1, Y1 = np.meshgrid(x, y)
Z1p = fzp(X1, Y1)
Z1n = fzn(X1, Y1)

X2, Z2 = np.meshgrid(x, z)
Y2p = fyp(X2, Z2)
Y2n = fyn(X2, Z2)

Y3, Z3 = np.meshgrid(y, z)
X3p = fxp(Y3, Z3)
X3n = fxn(Y3, Z3)

#fig = plt.figure()
ax2=fig.add_subplot(122, aspect='auto', adjustable='datalim',projection ='3d')
ax2.plot_wireframe(X1, Y1, Z1p, color ='orange', alpha=0.15)
ax2.plot_wireframe(X1, Y1, Z1n, color ='orange', alpha=0.15)

ax2.plot_wireframe(X2, Y2p, Z2, color ='orange', alpha=0.15)
ax2.plot_wireframe(X2, Y2n, Z2, color ='orange', alpha=0.15)

ax2.plot_wireframe(X3p, Y3, Z3, color ='orange', alpha=0.15)
ax2.plot_wireframe(X3n, Y3, Z3, color ='orange', alpha=0.15)


ax2.scatter(pointx, pointy, pointz, color = 'r',alpha=0.5)
ax2.scatter(xstrengths0, ystrengths0,zstrengths0, color='b', alpha=1)#, color ='orange')


ax2.plot(xlinex,[0,0],[0,0])
ax2.plot([0,0],yliney,[0,0])
ax2.plot([0,0],[0,0],zlinez)
ax2.set_zlim(-200,200)
ax2.set_xlim(-600,600)
ax2.set_ylim(-600,600)

ax2.set_xlabel(r'x failure stress, $\sigma_x$ (GPa)', fontsize=11)
ax2.set_ylabel(r'y failure  stress, $\sigma_y$ (GPa)', fontsize=11)
ax2.set_zlabel(r'z failure  stress, $\sigma_z$ (GPa)', fontsize=11)


ax2.set_title('Hills failure Envelope', fontsize=12);
plt.savefig('26hills_3D.png')
plt.show()
plt.close()

#plt.plot(index,TWtest)
#plt.show()
#plt.close()


####################################################### envelope plot
################## for 1'-2' ###################

sx= np.linspace(-300,200, 500)
sy= np.linspace(-350,200, 500)

x, y = np.meshgrid(sx, sy)
d=np.array(F11*x**2+F1*x+F22*y**2+F2*y+2*F12*x*y)

dh1=np.array(A*y**2+B*x**2+C*(x-y)**2)

#a=0.5*np.ones(np.shape(d),dtype=float)
for i in range(np.size(d,0)):
	for j in range(np.size(d,1)):
		if d[i][j]>2:
			d[i][j]=2
#			a[i][j]=0
		if dh1[i][j]>2:
			dh1[i][j]=2

a=0.5*np.ones(np.shape(d),dtype=float)

a=F22
b=F2+2*F12*sx
c=F11*sx**2 + F1*sx - 1
det=b**2-4*a*c
#print(det)

sya= (-b+(det)**0.5)/(2*a)
syb= (-b-(det)**0.5)/(2*a)


fig = plt.figure(figsize=(21.36,13.28))
ax=fig.add_subplot(121, aspect='auto', adjustable='datalim',projection ='3d')
#ax = plt.axes(projection ='3d')
p = ax.plot_surface(x, y, d, cmap=cm.coolwarm, alpha=0.5)
q = ax.plot(sx, sya, zs=1, zdir='z')
r = ax.plot(sx, syb, zs=1, zdir='z')
fig.colorbar(p, shrink=0.2, aspect=10)
ax.set_xlim3d([-300,200])
ax.set_ylim3d([-350,200])
ax.set_zlim3d([-0.5,2])
ax.set_xlabel(r'x stress, $\sigma_x$ (GPa)', fontsize=11)
ax.set_ylabel(r'y stress, $\sigma_y$ (GPa)', fontsize=11)
ax.set_title('tsai wu values in x-y plane', fontsize=12);



ax3=fig.add_subplot(122, aspect='auto', adjustable='datalim',projection ='3d')
#ax = plt.axes(projection ='3d')
p3 = ax3.plot_surface(x, y, dh1, cmap=cm.coolwarm, alpha=0.5)
#q3 = ax3.plot(sx, sya, zs=1, zdir='z')
#r3 = ax3.plot(sx, syb, zs=1, zdir='z')
fig.colorbar(p3, shrink=0.2, aspect=10)
ax3.set_xlim3d([-300,200])
ax3.set_ylim3d([-350,200])
ax3.set_zlim3d([-0.5,2])
ax3.set_xlabel(r'x stress, $\sigma_x$ (GPa)', fontsize=11)
ax3.set_ylabel(r'y stress, $\sigma_y$ (GPa)', fontsize=11)
ax3.set_title('hills 1 values in x-y plane', fontsize=12);



plt.savefig('xy criteria26 .png')
plt.show()

plt.close()


#plt.plot(sx,sya,'b')
#plt.plot(sx,syb,'b')

#plt.xlim([-850,150])
#plt.ylim([-700,150])

#plt.xlabel(r'Failure stress $\sigma_x$ (GPa)')

#plt.ylabel(r'Failure stress $\sigma_y$ (GPa)')
#plt.gca().set_aspect('equal',adjustable='box')
#plt.savefig('x-y envelopeb20.png')
#plt.show()
#plt.close()




####################################################### envelope plot
################## for 1'-3 ###################

sx= np.linspace(-450,200, 500)
sz= np.linspace(-200,50, 500)

x, z = np.meshgrid(sx, sz)
d=np.array(F11*x**2+F1*x+F33*z**2+F3*z+2*F13*x*z)

dh1=np.array(A*z**2+C*x**2+B*(x-z)**2)
#a=0.5*np.ones(np.shape(d),dtype=float)
for i in range(np.size(d,0)):
	for j in range(np.size(d,1)):
		if d[i][j]>2:
			d[i][j]=2
#			a[i][j]=0

		if dh1[i][j]>2:
			dh1[i][j]=2

a=0.5*np.ones(np.shape(d),dtype=float)

a=F33
b=F3+2*F13*sx
c=F11*sx**2 + F1*sx - 1
det=b**2-4*a*c
#print(det)

sza= (-b+(det)**0.5)/(2*a)
szb= (-b-(det)**0.5)/(2*a)

fig = plt.figure(figsize=(21.36,13.28))
ax=fig.add_subplot(121, aspect='auto', adjustable='datalim',projection ='3d')
#ax = plt.axes(projection ='3d')
p = ax.plot_surface(x, z, d, cmap=cm.coolwarm, alpha=0.5)
q = ax.plot(sx, sza, zs=1, zdir='z')
r = ax.plot(sx, szb, zs=1, zdir='z')
fig.colorbar(p, shrink=0.2, aspect=10)
ax.set_xlim3d([-450,200])
ax.set_ylim3d([-200,50])
ax.set_zlim3d([-3.5,2])
ax.set_xlabel(r'x stress, $\sigma_x$ (GPa)', fontsize=11)
ax.set_ylabel(r'z stress, $\sigma_z$ (GPa)', fontsize=11)
ax.set_title('tsai wu values in x-z plane', fontsize=12);


ax3=fig.add_subplot(122, aspect='auto', adjustable='datalim',projection ='3d')
#ax = plt.axes(projection ='3d')
p3 = ax3.plot_surface(x, z, dh1, cmap=cm.coolwarm, alpha=0.5)
#q3 = ax3.plot(sx, sya, zs=1, zdir='z')
#r3 = ax3.plot(sx, syb, zs=1, zdir='z')
fig.colorbar(p3, shrink=0.2, aspect=10)
ax3.set_xlim3d([-450,200])
ax3.set_ylim3d([-200,50])
ax3.set_zlim3d([-0.5,2])
ax3.set_xlabel(r'x stress, $\sigma_x$ (GPa)', fontsize=11)
ax3.set_ylabel(r'z stress, $\sigma_z$ (GPa)', fontsize=11)
ax3.set_title('hills 1 values in x-z plane', fontsize=12);

plt.savefig('xz criteria26 .png')
plt.show()

plt.close()


####################################################### envelope plot
################## for 2'-3 ###################

sy= np.linspace(-500,250, 500)
sz= np.linspace(-200,50, 500)

y, z = np.meshgrid(sy, sz)
d=np.array(F22*y**2+F2*y+F33*z**2+F3*z+2*F23*y*z)

dh1=np.array(B*z**2+C*y**2+A*(y-z)**2)		#a=0.5*np.ones(np.shape(d),dtype=float)
for i in range(np.size(d,0)):
	for j in range(np.size(d,1)):
		if d[i][j]>2:
			d[i][j]=2
#			a[i][j]=0

		if dh1[i][j]>2:
			dh1[i][j]=2

a=0.5*np.ones(np.shape(d),dtype=float)

a=F33
b=F3+2*F23*sy
c=F22*sy**2 + F2*sy - 1
det=b**2-4*a*c
#print(det)

sza= (-b+(det)**0.5)/(2*a)
szb= (-b-(det)**0.5)/(2*a)

fig = plt.figure(figsize=(21.36,13.28))
ax=fig.add_subplot(121, aspect='auto', adjustable='datalim',projection ='3d')
#ax = plt.axes(projection ='3d')
p = ax.plot_surface(y, z, d, cmap=cm.coolwarm, alpha=0.5)
q = ax.plot(sy, sza, zs=1, zdir='z')
r = ax.plot(sy, szb, zs=1, zdir='z')
fig.colorbar(p, shrink=0.2, aspect=10)
ax.set_xlim3d([-500,250])
ax.set_ylim3d([-200,50])
ax.set_zlim3d([-3.5,2])
ax.set_xlabel(r'y stress, $\sigma_y$ (GPa)', fontsize=11)
ax.set_ylabel(r'z stress, $\sigma_z$ (GPa)', fontsize=11)
ax.set_title('tsai wu values in y-z plane', fontsize=12);


ax3=fig.add_subplot(122, aspect='auto', adjustable='datalim',projection ='3d')
#ax = plt.axes(projection ='3d')
p3 = ax3.plot_surface(y, z, dh1, cmap=cm.coolwarm, alpha=0.5)
#q3 = ax3.plot(sx, sya, zs=1, zdir='z')
#r3 = ax3.plot(sx, syb, zs=1, zdir='z')
fig.colorbar(p3, shrink=0.2, aspect=10)
ax3.set_xlim3d([-500,250])
ax3.set_ylim3d([-200,50])
ax3.set_zlim3d([-0.5,2])
ax3.set_xlabel(r'y stress, $\sigma_y$ (GPa)', fontsize=11)
ax3.set_ylabel(r'z stress, $\sigma_z$ (GPa)', fontsize=11)
ax3.set_title('hills 1 values in y-z plane', fontsize=12);

plt.savefig('yz criteria26 .png')
plt.show()

plt.close()


#######################################






