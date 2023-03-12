# %% 
import sys,glob,os 
import scipy.io
import numpy as np

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
import Analysis.neural.src.rf_fit as rf 
from Admin.csv_queryExp import load_data

subject = 'FT008'
probe = 'probe0'
data_dict = {'events':{'_av_trials':['squareAzimuth','squareElevation','squareOnTimes']},probe:{'spikes':['times','clusters','depths','_av_shankIDs']}}
recordings = load_data(subject = subject, expDate = '2021-01-15',data_name_dict=data_dict,expDef='sparseNoise')


# %%
# try finding receptive fields in any recording...
findRF = rf.rf_fit()
shankpos = np.zeros((4,3)) 
myRFs={}
for idx,rec in recordings.iterrows():
    if (shankpos.sum(axis=1)==0).any():
        try:
            spikes = rec[probe].spikes
            sn_info = rec.events._av_trials

            findRF.add_sparseNoise_info(sn_info)
            findRF.bin_spikes_per_loc(spikes)
            a =findRF.binned_spikes_depths['array']
            findnonzeros = np.where(a.sum(axis=2)>a.sum(axis=2).max()*0.01) # check which pieces have nonzero activity
            shnks = np.unique(findnonzeros[0]) # which shanks have nonzero activity
            # now how do yo calculate wihch shankpos corresponds to which startpos....
            for shk in shnks:
                depths_idx = findnonzeros[1][findnonzeros[0]==shk] # on that shank, which depths 
                is_ventricle_start = np.append(np.diff(depths_idx)>1,False)
                startpos= depths_idx[is_ventricle_start]           
                # if this shank has not been calculated already...
                if shankpos[shk,:].sum()==0:
                    foundrf = 0                    
                    RF_pos = []
                    resps = []
                    for p0 in startpos:
                        ct=0
                        while (foundrf<5 and (ct<4)): # search for receptive fields
                            r =findRF.get_response_binned(a[shk,p0-ct,:]) #8.8s for a single one... # now imagine 
                            mf = findRF.fit_predict(r)
                            depth_um = findRF.binned_spikes_depths['depthscale'][int(p0-ct)]
                            if mf is not np.nan:
                                RF_pos.append(np.append(depth_um,mf))
                                resps.append(r)
                                foundrf+=1
                            ct+=1

                    if foundrf>0:
                        myRFs[shk]=resps
                        RFpos = np.array(RF_pos)
                        print(shk,RFpos[:,0])
                        SCsurface = np.max(RFpos[:,0])
                        xcenter=np.mean(RFpos[:,2])
                        ycenter = np.mean(RFpos[:,4])
                        xcenter_deg=xcenter*7.5-135
                        ycenter_deg=75-7.5*ycenter-41.25

                        shankpos[shk,:] = [SCsurface, xcenter_deg,ycenter_deg]
        except:
            print('did not find RF on  %s' % rec.expFolder)


# %%
# so sometimes it is the case that there is no RF on one of the shanks 
# one can still extapolate 
is_fitted = (shankpos[:,0]>0)
if np.sum(is_fitted)==4: 
    print('good job. You got receptive fields on all shanks')    
elif np.sum(is_fitted)==0: 
    print('Either you have some error, or you dont have RFs. You suck.')
else:
    print('trying to interpolate your missing shank loc....')
    print('you might need to check for this as I did not account for all possible scenarios.')
    shank_to_correct = np.where(~is_fitted)[0]
    if (shank_to_correct.size==1):
        if shank_to_correct==0:
            print('g')
            shankpos[0,:] = shankpos[1,:] -np.mean(np.diff(shankpos[1:,:],axis=0),axis=0)
        elif (shank_to_correct==1): 
            shankpos[1,:] = (shankpos[0,:]+shankpos[2,:])/2
        elif (shank_to_correct==2): 
            shankpos[2,:] = (shankpos[1,:]+shankpos[3,:])/2
        elif shank_to_correct==3: 
            shankpos[3,:] = shankpos[2,:] +np.mean(np.diff(shankpos[:-1,:],axis=0),axis=0)

    if (shank_to_correct.size==2): 
        if (shank_to_correct==np.array([0,1])).all():
            shankpos[1,:] = shankpos[2,:] + (shankpos[2,:]-shankpos[3,:])
            shankpos[0,:] = shankpos[2,:] + (shankpos[2,:]-shankpos[3,:])
print(shankpos)

# %%
print('saving shankpos...')
savepath = r'C:\Users\Flora\Documents\Processed data\Audiovisual\%s' % (subject)
if not os.path.exists(savepath):
    os.makedirs(savepath)
np.save(r'%s\%s_SC_shank_pos.npy' % (savepath,probe),shankpos)


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import utils.plotting as pp
fitted_shanks= np.where(is_fitted)[0]

fig = plt.figure()
gs = GridSpec(1, fitted_shanks.size)

for ix,sh in enumerate(fitted_shanks): 
    ax = fig.add_subplot(gs[ix])
    resp = myRFs[sh][0]
    vmax = np.max(np.abs(resp))/1.3
    plt.imshow(resp,cmap='coolwarm',vmin=-vmax,vmax=vmax)
    ax.set_title('shank %0.d' % sh)
    pp.off_axes(ax)
# save figure 

fig.savefig(r'%s\%s_receptive_fields.png' % (savepath,probe),bbox_inches='tight')
# %%
