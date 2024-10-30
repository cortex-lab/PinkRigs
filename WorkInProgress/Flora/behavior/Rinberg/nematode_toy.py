#%%
import numpy as np 
import random
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs\WorkInProgress\Flora") 
print(sys.path)

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from behavior.glm.plot_utils import copy_svg_to_clipboard



def get_evidence(stim,inactivated_layer = -1,*biases):

    std = .1
    total = stim # 1st layer 
  #  print('getSample')
    for i,b in enumerate(biases):
        if i==inactivated_layer: 
            total*= 0.4
        else:
            total+=random.gauss(b,std)
 #       print(total)

    return total


def get_choice(visL,visR,oL,oR,bL,bR):
    visL_ = get_evidence(visL,oL,*bL)
    visR_ = get_evidence(visR,oR,*bR)
    choice = int(visL_ <= visR_)
    return choice



# simulate the mmodel with different biases
n_samples = 10000
contrasts = [0,0.1,0.2,0.4,0.6,0.8,1]



visL = np.concatenate((
            np.random.choice(contrasts,n_samples),
            np.zeros(n_samples)),
            axis = 0
) 

visR = np.concatenate((
            np.zeros(n_samples),
            np.random.choice(contrasts,n_samples)),
            axis = 0
) 


mu = .1
n_layers = 10
bL =(mu,) * n_layers
bR = (mu,) * n_layers

# Get the colors from the viridis colormap


fig,ax = plt.subplots(1,1,figsize=(3.5,4.2))
colors = plt.cm.cividis(np.linspace(0, .8, n_layers))

slopes,intercepts = [],[]
for i in range(n_layers):

    choices = np.array([get_choice(l,r,i-1,i-1,bL,bR) for l,r in zip(visL,visR)])

    m = LogisticRegression()
    m.fit(np.array((visL,visR)).T,choices)
    slopes.append(m.coef_)
    intercepts.append(m.intercept_)
    
    visDiff = visR-visL
    unique_visDiff = np.unique(visDiff)
    pR = np.array([np.mean(choices[visDiff==v]) for v in unique_visDiff])

    if i==0:
        ax.plot(unique_visDiff,np.log((pR/(1-pR))),color='k',lw=5)
    else:
        if i % 2 == 0:
            ax.plot(unique_visDiff,np.log((pR/(1-pR))),color=colors[i])
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')

plt.tight_layout()
ax.set_ylim([-5,8])
plt.show()

    # 
copy_svg_to_clipboard(fig)
# %%

fig,(slopeax,biasax) = plt.subplots(1,2,figsize=(3.6,1.8))
ss  = np.concatenate(slopes)
bb = np.concatenate(intercepts)
slopeax.plot(ss[:,0] - ss[0,0])
slopeax.set_xlabel('layer inactivated')
slopeax.set_ylabel('slope change')

biasax.plot(bb-bb[0])
biasax.set_xlabel('layer # inactivated')
biasax.set_ylabel('bias change')

slopeax.set_ylim([0,2.1])
biasax.set_ylim([-.1,7])
fig.suptitle('bilateral')

copy_svg_to_clipboard(fig)

# %%
