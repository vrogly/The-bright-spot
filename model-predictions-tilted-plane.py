import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

""" For Linux install """
import matplotlib.font_manager as fm  # Fixing font

fe = fm.FontEntry(
    fname="/usr/share/fonts/cm-unicode/cmunrm.otf",  # Path to custom installed font (in this case computer modern roman)
    name="cmunrm-manual",
)
fm.fontManager.ttflist.insert(0, fe)  # or append is fine
plt.rcParams["font.family"] = fe.name
plt.rcParams["font.size"] = 24
plt.rcParams["mathtext.fontset"] = "cm"
##### END PREAMBLE #####

w = 1.5e-3 # mm
xRes = 100
nn= 4
xRange = np.linspace(0.001,2.5,xRes)
dataMat0 = np.zeros((nn,xRes))
dataMat45 = np.zeros((nn,xRes))
lamb = 532e-9 # m
delta = w

def deltaL (y,xD,n,delta):
	Ltop = delta + np.sqrt((w/2-y)**2 + (xD-delta)**2)
	Lbot = np.sqrt((w/2+y)**2 + xD**2)
	return lamb*n-(Ltop-Lbot)


for n in range(nn):
	yVals0 = []
	yVals45 = []
	for x in xRange:
		yVals0 += [ fsolve(deltaL, x0=0,args=(x,n,0))[0] ]
		yVals45 += [ fsolve(deltaL, x0=0,args=(x,n,delta))[0] ]
	dataMat0[n,:] = np.array(yVals0)
	dataMat45[n,:] = np.array(yVals45)
fig = plt.figure(figsize=(11.69,8.27))
ax = plt.axes()

ax.set(ylim=(-0.01,0.5))

for i in range(nn):
	plt.plot(xRange,np.abs(dataMat0[i,:])/w,label=f"Flat: $\delta = 0$, $n={i}$")
	plt.plot(xRange,np.abs(dataMat45[i,:])/w,label=f"$45^\circ$: $\delta = w$, $n={i}$",linestyle='--')
ax.set_xlabel('$x_D$ (m)')
ax.set_ylabel('$y_D/w$')
ax.set_xscale('log')
plt.legend()
plt.show()
fig.savefig('x-vs-ydiff-45.pdf', format='pdf',dpi=800,bbox_inches='tight')
