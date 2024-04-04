#%%
import pyddm
import pyddm.plot
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12

import glob,sys
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))
from Analysis.pyutils.plotting import off_axes




class DriftOnly(pyddm.Drift):
    name = "drift"
    required_parameters = ["vis","bias"]
    required_conditions = ["C"]
    def get_drift(self, conditions, **kwargs):
        return (self.vis * conditions["C"]) + self.bias

vis_coef = 8
noise = 2
bound = 1
x0 =0
bias = 0

x0s = np.linspace(-.7,.7,5) 
vis_contrasts = np.linspace(-1,1,40)
bounds = np.linspace(1,1.5,5)
vis_coefs = np.linspace(8,16,5)
biases = np.linspace(0,10,5)
noises = np.linspace(1,3,5)

#noises = noise * bounds

noises  = (1/(np.abs(x0s)+1)) * noise

def get_rt_quartiles(m,v,correct_only = True):
    sol = m.solve(conditions={"C":v})

    if correct_only:
        correct_side = np.sign(v)
    else:
        correct_side = 0 # when we want to get both

    percentiles = [.25,.5,.75]
    l = [np.interp(np.ptp(sol.cdf('Left'))*p,sol.cdf('Left'),sol.t_domain) for p in percentiles]
    r = [np.interp(np.ptp(sol.cdf('Right'))*p,sol.cdf('Right'),sol.t_domain) for p in percentiles]
    j = [np.interp(np.ptp(sol.cdf('Right')+sol.cdf('Left'))*p,sol.cdf('Right')+sol.cdf('Left'),sol.t_domain) for p in percentiles]

    if correct_side==-1: 
        lower,mid,upper = l
    elif correct_side==1: 
        lower,mid,upper = r
    elif correct_side==0: 
        lower,mid,upper = j

    return mid


fig,(axp,ax1,ax2,axc,axc1) = plt.subplots(1,5,figsize=(15,3))
colors = plt.cm.inferno(np.linspace(0.2,.8,x0s.size))
for i,bound in enumerate(bounds):
    m = pyddm.Model(drift=DriftOnly(vis=vis_coef,bias=bias),
                    noise=pyddm.NoiseConstant(noise=noise),
                    bound=pyddm.BoundConstant(B=bound),
                    overlay=pyddm.OverlayChain(overlays=[
                        pyddm.OverlayNonDecision(nondectime=.3),
                        pyddm.OverlayExponentialMixture(pmixturecoef=0,
                        rate=1),
                        ]),
                    IC=pyddm.ICPoint(x0=x0),
                    dt=.01, dx=.01, T_dur=1.5,choice_names = ('Right','Left'))


    pdf_r = m.solve(conditions={"C":0}).pdf('Right')
    axp.plot(m.t_domain(),(pdf_r-np.min(pdf_r))/(np.max(pdf_r) - np.min(pdf_r)),color=colors[i])
    #axp.plot(m.t_domain(),(-1)*m.solve(conditions={"C":0}).cdf('Left'),color=colors[i])


    psychometric = np.array([m.solve(conditions={"C": v}).prob('Right') for v in vis_contrasts])
    psychometric_log = np.log10(psychometric/(1-psychometric))

    ax1.plot(vis_contrasts,psychometric_log,color=colors[i])

    ax2.plot(vis_contrasts,psychometric,color=colors[i])
    # chronometric
    #chronometric = ([m.solve(conditions={"C":v}).mean_decision_time() for v in vis_contrasts])
    chronometric_c = [get_rt_quartiles(m,v,correct_only=False) for v in vis_contrasts]
    chronometric_e = [get_rt_quartiles(m,v,correct_only=True) for v in vis_contrasts]

    axc.plot(vis_contrasts,chronometric_c,color=colors[i])
    axc1.plot(vis_contrasts,chronometric_e,color=colors[i])


ax1.axhline(0,color='k',linestyle='--')
ax1.axvline(0,color='k',linestyle='--')

axp.set_xlabel('time (s)')
axp.set_ylabel('pdf at 0 contrast')

ax1.set_xlabel('contrast')
ax1.set_ylabel('log odds')

ax2.set_xlabel('contrast')
ax2.set_ylabel('p(R)')

axc.set_xlabel('contrast')
axc.set_ylabel('median reaction time on correct')

axc1.set_xlabel('contrast')
axc1.set_ylabel('median reaction time on incorrect') 

plt.show()

# %%


vis_coef = 8
noise = 2
bound = 1
x0 =0
bias = 0

x0s = np.linspace(-.7,.7,5) 
vis_contrasts = np.linspace(-1,1,40)
bounds = np.linspace(1,1.5,5)
vis_coefs = np.linspace(8,16,5)
biases = np.linspace(0,10,5)
noises = np.linspace(1,3,5)

# simulation plot for the poster
# i.e. uppper and lower 
fig,axu= plt.subplots(1,1,figsize=(4,4))
colors = plt.cm.inferno(np.linspace(0.2,.8,x0s.size))
for i,bound in enumerate(bounds):
    m = pyddm.Model(drift=DriftOnly(vis=vis_coef,bias=bias),
                    noise=pyddm.NoiseConstant(noise=noise),
                    bound=pyddm.BoundConstant(B=bound),
                    overlay=pyddm.OverlayChain(overlays=[
                        pyddm.OverlayNonDecision(nondectime=.3),
                        pyddm.OverlayExponentialMixture(pmixturecoef=0,
                        rate=1),
                        ]),
                    IC=pyddm.ICPoint(x0=x0),
                    dt=.01, dx=.01, T_dur=2,choice_names = ('Right','Left'))


    pdf_r = m.solve(conditions={"C":0}).pdf('Right')
    axu.plot(m.t_domain(),(pdf_r-np.min(pdf_r))/(np.max(pdf_r) - np.min(pdf_r)),color=colors[i])

off_axes(axu)


mypath = r'C:\Users\Flora\Pictures\SfN2023'
savename = mypath + '\\' + 'simulation_bound.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

    #axu.plot(m.t_domain(),pdf_r,color=colors[i])
    #pdf_r = m.solve(conditions={"C":0}).pdf('Left')
    #axl.plot(m.t_domain(),-(pdf_r-np.min(pdf_r))/(np.max(pdf_r) - np.min(pdf_r)),color=colors[i])
    #axl.plot(m.t_domain(),-pdf_r,color=colors[i])

# %%
