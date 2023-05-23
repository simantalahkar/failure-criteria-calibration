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
#######################################################################################

tx=[]
ty=[]
tz=[]


sx1=104.4
sy1=0
sz1=-10			#done

tx.append(sx1)
ty.append(sy1)
tz.append(sz1)

sx3=118.2
sy3=0
sz3=-65.7		#done

tx.append(sx3)
ty.append(sy3)
tz.append(sz3)


sx37=-445.8
sy37=-384.4
sz37=-258		#done

tx.append(sx37)
ty.append(sy37)
tz.append(sz37)


sx36=-391.7
sy36=-454.5
sz36=-260.5		#done

tx.append(sx36)
ty.append(sy36)
tz.append(sz36)


sx35=112.5
sy35=-113.9
sz35=-102.9		#done

tx.append(sx35)
ty.append(sy35)
tz.append(sz35)

sx34=-138
sy34=126.3
sz34=-103.9		#done

tx.append(sx34)
ty.append(sy34)
tz.append(sz34)

sx33=-262
sy33=0
sz33=-156.6		#done

tx.append(sx33)
ty.append(sy33)
tz.append(sz33)

sx32=94.5
sy32=94.5
sz32=-64.2		#done

tx.append(sx32)
ty.append(sy32)
tz.append(sz32)


sx31=80.8
sy31=10.7
sz31=1.6	

tx.append(sx31)
ty.append(sy31)
tz.append(sz31)

sx7=88.3
sy7=11.7
sz7=0			#done

tx.append(sx7)
ty.append(sy7)
tz.append(sz7)

sx8=14.2
sy8=100.4
sz8=0			#done

tx.append(sx8)
ty.append(sy8)
tz.append(sz8)

sx70=90.8		#done
sy70=0
sz70=0

tx.append(sx70)
ty.append(sy70)
tz.append(sz70)

sx80=0
sy80=104.5
sz80=0			#done

tx.append(sx80)
ty.append(sy80)
tz.append(sz80)

sx20=0
sy20=0
sz20=-132.9		#done

tx.append(sx20)
ty.append(sy20)
tz.append(sz20)

sx10=-241.6
sy10=0
sz10=-123.7		#done

#tx.append(sx10)				############################ deviation
#ty.append(sy10)
#tz.append(sz10)

sx11=0
sy11=-262
sz11=-156.6		#done

tx.append(sx11)
ty.append(sy11)
tz.append(sz11)

sx12=126.2		#
sy12=-50
sz12=-92.1

tx.append(sx12)
ty.append(sy12)
tz.append(sz12)

sx13=-324.6		
sy13=-455.6
sz13=-249.9

tx.append(sx13)
ty.append(sy13)
tz.append(sz13)

sx14=-52.8
sy14=142
sz14=-85.8

tx.append(sx14)
ty.append(sy14)
tz.append(sz14)

sx15=46.4
sy15=77
sz15=1.4

tx.append(sx15)
ty.append(sy15)
tz.append(sz15)

sx16=85.2
sy16=100
sz16=-65.7

tx.append(sx16)
ty.append(sy16)
tz.append(sz16)

sx17=-456.4
sy17=-514.8
sz17=-311.8

#tx.append(sx17)					################################## deviation
#ty.append(sy17)
#tz.append(sz17)

sx18=-440
sy18=-440
sz18=-265.8

tx.append(sx18)
ty.append(sy18)
tz.append(sz18)

sx19=66.6
sy19=66.6
sz19=1.3

tx.append(sx19)
ty.append(sy19)
tz.append(sz19)

sx2=-435.4
sy2=-309.4
sz2=-244.6

#tx.append(sx2)
#ty.append(sy2)
#tz.append(sz2)

sx29=12.6
sy29=89.6
sz29=1.6

tx.append(sx29)
ty.append(sy29)
tz.append(sz29)

sx28=112.7
sy28=16.9
sz28=-82

tx.append(sx28)
ty.append(sy28)
tz.append(sz28)

sx27=20.2
sy27=127.8
sz27=-74.2

tx.append(sx27)
ty.append(sy27)
tz.append(sz27)

sx26=-334.4
sy26=-77.4
sz26=-184.6

tx.append(sx26)
ty.append(sy26)
tz.append(sz26)

sx25=-5.6
sy25=134.8
sz25=-79

tx.append(sx25)
ty.append(sy25)
tz.append(sz25)


sx24=117.7
sy24=-6.9
sz24=-86.1

tx.append(sx24)
ty.append(sy24)
tz.append(sz24)


sx23=-415.3
sy23=-229.2
sz23=-227.8

tx.append(sx23)
ty.append(sy23)
tz.append(sz23)

sx22=-380.2
sy22=-148.5
sz22=-207.2

tx.append(sx22)
ty.append(sy22)
tz.append(sz22)

sx21=-402.4
sy21=-196.4
sz21=-219.8

tx.append(sx21)
ty.append(sy21)
tz.append(sz21)


cxx2=sx2**2
cx2=sx2
cxy2=2*sx2*sy2
cyy2=sy2**2
cy2=sy2
cyz2=2*sy2*sz2
czz2=sz2**2
cz2=sz2
cxz2=2*sx2*sz2



cxx1=sx1**2
cx1=sx1
cxy1=2*sx1*sy1
cyy1=sy1**2
cy1=sy1
cyz1=2*sy1*sz1
czz1=sz1**2
cz1=sz1
cxz1=2*sx1*sz1

cxx19=sx19**2
cx19=sx19
cxy19=2*sx19*sy19
cyy19=sy19**2
cy19=sy19
cyz19=2*sy19*sz19
czz19=sz19**2
cz19=sz19
cxz19=2*sx19*sz19

cxx18=sx18**2
cx18=sx18
cxy18=2*sx18*sy18
cyy18=sy18**2
cy18=sy18
cyz18=2*sy18*sz18
czz18=sz18**2
cz18=sz18
cxz18=2*sx18*sz18

cxx17=sx17**2
cx17=sx17
cxy17=2*sx17*sy17
cyy17=sy17**2
cy17=sy17
cyz17=2*sy17*sz17
czz17=sz17**2
cz17=sz17
cxz17=2*sx17*sz17

cxx16=sx16**2
cx16=sx16
cxy16=2*sx16*sy16
cyy16=sy16**2
cy16=sy16
cyz16=2*sy16*sz16
czz16=sz16**2
cz16=sz16
cxz16=2*sx16*sz16

cxx10=sx10**2
cx10=sx10
cxy10=2*sx10*sy10
cyy10=sy10**2
cy10=sy10
cyz10=2*sy10*sz10
czz10=sz10**2
cz10=sz10
cxz10=2*sx10*sz10

cxx11=sx11**2
cx11=sx11
cxy11=2*sx11*sy11
cyy11=sy11**2
cy11=sy11
cyz11=2*sy11*sz11
czz11=sz11**2
cz11=sz11
cxz11=2*sx11*sz11

cxx12=sx12**2
cx12=sx12
cxy12=2*sx12*sy12
cyy12=sy12**2
cy12=sy12
cyz12=2*sy12*sz12
czz12=sz12**2
cz12=sz12
cxz12=2*sx12*sz12

cxx13=sx13**2
cx13=sx13
cxy13=2*sx13*sy13
cyy13=sy13**2
cy13=sy13
cyz13=2*sy13*sz13
czz13=sz13**2
cz13=sz13
cxz13=2*sx13*sz13

cxx14=sx14**2
cx14=sx14
cxy14=2*sx14*sy14
cyy14=sy14**2
cy14=sy14
cyz14=2*sy14*sz14
czz14=sz14**2
cz14=sz14
cxz14=2*sx14*sz14

cxx15=sx15**2
cx15=sx15
cxy15=2*sx15*sy15
cyy15=sy15**2
cy15=sy15
cyz15=2*sy15*sz15
czz15=sz15**2
cz15=sz15
cxz15=2*sx15*sz15


cxx20=sx20**2
cx20=sx20
cxy20=2*sx20*sy20
cyy20=sy20**2
cy20=sy20
cyz20=2*sy20*sz20
czz20=sz20**2
cz20=sz20
cxz20=2*sx20*sz20

cxx3=sx3**2
cx3=sx3
cxy3=2*sx3*sy3
cyy3=sy3**2
cy3=sy3
cyz3=2*sy3*sz3
czz3=sz3**2
cz3=sz3
cxz3=2*sx3*sz3

cxx33=sx33**2
cx33=sx33
cxy33=2*sx33*sy33
cyy33=sy33**2
cy33=sy33
cyz33=2*sy33*sz33
czz33=sz33**2
cz33=sz33
cxz33=2*sx33*sz33

cxx34=sx34**2
cx34=sx34
cxy34=2*sx34*sy34
cyy34=sy34**2
cy34=sy34
cyz34=2*sy34*sz34
czz34=sz34**2
cz34=sz34
cxz34=2*sx34*sz34

cxx35=sx35**2
cx35=sx35
cxy35=2*sx35*sy35
cyy35=sy35**2
cy35=sy35
cyz35=2*sy35*sz35
czz35=sz35**2
cz35=sz35
cxz35=2*sx35*sz35

cxx32=sx32**2
cx32=sx32
cxy32=2*sx32*sy32
cyy32=sy32**2
cy32=sy32
cyz32=2*sy32*sz32
czz32=sz32**2
cz32=sz32
cxz32=2*sx32*sz32


cxx7=sx7**2
cx7=sx7
cxy7=2*sx7*sy7
cyy7=sy7**2
cy7=sy7
cyz7=2*sy7*sz7
czz7=sz7**2
cz7=sz7
cxz7=2*sx7*sz7

cxx8=sx8**2
cx8=sx8
cxy8=2*sx8*sy8
cyy8=sy8**2
cy8=sy8
cyz8=2*sy8*sz8
czz8=sz8**2
cz8=sz8
cxz8=2*sx8*sz8

cxx70=sx70**2
cx70=sx70
cxy70=2*sx70*sy70
cyy70=sy70**2
cy70=sy70
cyz70=2*sy70*sz70
czz70=sz70**2
cz70=sz70
cxz70=2*sx70*sz70

cxx80=sx80**2
cx80=sx80
cxy80=2*sx80*sy80
cyy80=sy80**2
cy80=sy80
cyz80=2*sy80*sz80
czz80=sz80**2
cz80=sz80
cxz80=2*sx80*sz80


strengths=np.array([[cxx32, cx32, cxy32, cyy32, cy32, cyz32, czz32, cz32, cxz32], \
	[cxx70, cx70, cxy70, cyy70, cy70, cyz70, czz70, cz70, cxz70], \
	[cxx20, cx20, cxy20, cyy20, cy20, cyz20, czz20, cz20, cxz20],\
	[cxx80, cx80, cxy80, cyy80, cy80, cyz80, czz80, cz80, cxz80], \
	[cxx11, cx11, cxy11, cyy11, cy11, cyz11, czz11, cz11, cxz11],\
	[cxx2, cx2, cxy2, cyy2, cy2, cyz2, czz2, cz2, cxz2],\
	[cxx19, cx19, cxy19, cyy19, cy19, cyz19, czz19, cz19, cxz19],\
	[cxx13, cx13, cxy13, cyy13, cy13, cyz13, czz13, cz13, cxz13],\
	[cxx34, cx34, cxy34, cyy34, cy34, cyz34, czz34, cz34, cxz34]])
	
xstrengths=np.array([sx14,sx70,sx20,sx80,sx13,sx32,sx12,sx11,sx2,sx19,sx33,sx34,sx35,sx10])
ystrengths=np.array([sy14,sy70,sy20,sy80,sy13,sy32,sy12,sy11,sy2,sy19,sy33,sy34,sy35,sy10])
zstrengths=np.array([sz14,sz70,sz20,sz80,sz13,sz32,sz12,sz11,sz2,sz19,sz33,sz34,sz35,sz10])

xstrengths0=np.array([sx11,sx70,sx20,sx80,sx2,sx13,sx19,sx32,sx34])
ystrengths0=np.array([sy11,sy70,sy20,sy80,sy2,sy13,sy19,sy32,sy34])
zstrengths0=np.array([sz11,sz70,sz20,sz80,sz2,sz13,sz19,sz32,sz34])

#print('1')
#print(tx)
#print(xstrengths0)

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






