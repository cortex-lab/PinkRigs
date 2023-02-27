import sys,os,pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import warnings # just because varance explained van be zero divisor (that's ok with no spike neuron etc)
warnings.filterwarnings("ignore")


from Admin.csv_queryExp import Bunch,load_ephys_independent_probes

import Analysis.neural.utils.spike_dat as su 
from Analysis.neural.utils.ev_dat import postactive
import Analysis.neural.utils.plotting as my_plotter

def sort_trialtypes_to_conditions(blank,vis,aud,MS,blank_reps=0,myContrast=1,mySPL=0.1,n_trials=None,align_type='vis'):
     
    params = {'visAzimuths': vis.azimuths.values,'audAzimuths': aud.azimuths.values}

    conditions = {'blank': blank_reps, 
                'vis':vis.azimuths.size,
                'aud': aud.azimuths.size, 
                'multi_congruent': MS.congruent_azimuths.size, 
                'multi_incongruent': MS.incongruent_vis_azimuths.size}


# for blanks I will write -1000 out of convenience -- nans are difficult to interpret and check against!!
    vis_azis = np.concatenate((np.ones(blank_reps)*-1000, 
                            vis.azimuths.values,np.ones(aud.azimuths.size)*-1000,
                            MS.congruent_azimuths[0,:],
                            MS.incongruent_vis_azimuths[0,:])) 

    aud_azis = np.concatenate((np.ones(blank_reps)*-1000, 
                            np.ones(vis.azimuths.size)*-1000,aud.azimuths.values,
                            MS.congruent_azimuths[0,:],
                            MS.incongruent_aud_azimuths[0,:])) 

    

    n_conditions=0
    for c in conditions.keys():
        n_conditions += conditions[c]
        
    n_params = 0 
    for p in params.keys():
        n_params += params[p].size

    if n_trials == None:
        n_trials = blank.trials.size

    indicator_matrix = np.zeros((n_conditions,n_params))

    trig_times_matrix = np.zeros((n_conditions,n_trials))

    c_idx=0
    for b in range(conditions['blank']):
        trig_times_matrix[c_idx,:]= blank.sel(trials=blank.trials.values,timeID='ontimes').values[:n_trials]    
        c_idx += 1

    for v in range(conditions['vis']):    
        myVazi = params['visAzimuths'][v]    
        vis_on = vis.sel(azimuths=myVazi,contrast=myContrast,trials=blank.trials.values,timeID='ontimes').values
        
        if align_type == 'aud':
            MS_aud=MS.sel(visazimuths=myVazi,audazimuths=myVazi,SPL=mySPL,contrast=myContrast,trials=blank.trials.values,timeID='audontimes').values
            MS_vis=MS.sel(visazimuths=myVazi,audazimuths=myVazi,SPL=mySPL,contrast=myContrast,trials=blank.trials.values,timeID='visontimes').values
            AVdiff=MS_aud-MS_vis
            vis_on=np.ravel(vis_on+AVdiff)

        trig_times_matrix[c_idx,:]= vis_on[:n_trials]

        p_idx = v
        indicator_matrix[c_idx,p_idx] =1 
        c_idx += 1
            
    for a in range(conditions['aud']):        
        myAazi = params['audAzimuths'][a] 
        aud_on=aud.sel(azimuths=myAazi,SPL=mySPL,trials=blank.trials.values,timeID='ontimes').values
        
        if align_type == 'vis':
            MS_aud=MS.sel(visazimuths=myAazi,audazimuths=myAazi,SPL=mySPL,contrast=myContrast,trials=blank.trials.values,timeID='audontimes').values
            MS_vis=MS.sel(visazimuths=myAazi,audazimuths=myAazi,SPL=mySPL,contrast=myContrast,trials=blank.trials.values,timeID='visontimes').values
            AVdiff=MS_aud-MS_vis
            aud_on=np.ravel(aud_on-AVdiff)


        trig_times_matrix[c_idx,:] = aud_on[:n_trials]
        p_idx = params['visAzimuths'].size + a
        indicator_matrix[c_idx,p_idx] =1 
        c_idx += 1  

    for m in range(conditions['multi_congruent']): 
        myMazi = params['visAzimuths'][m]
        if align_type == 'vis':
            congruent_times = MS.sel(visazimuths=myMazi,contrast=myContrast,audazimuths=myMazi,SPL=mySPL,trials=blank.trials.values,timeID='visontimes').values
        elif align_type == 'aud':
            congruent_times = MS.sel(visazimuths=myMazi,contrast=myContrast,audazimuths=myMazi,SPL=mySPL,trials=blank.trials.values,timeID='audontimes').values
        
        trig_times_matrix[c_idx,:]=congruent_times[:n_trials]
        #vis weights
        p_idx_v = m
        indicator_matrix[c_idx,p_idx_v]=1
        # aud weights
        p_idx_a = params['visAzimuths'].size + m
        indicator_matrix[c_idx,p_idx_a]=1
        c_idx += 1

    for m in range(conditions['multi_incongruent']):
        myAazi = int(MS.incongruent_aud_azimuths[:,m]) 
        myVazi = int(MS.incongruent_vis_azimuths[:,m])   
        
        if align_type == 'vis':
            incongruent_times = MS.sel(visazimuths=myVazi,contrast=myContrast,audazimuths=myAazi,SPL=mySPL,trials=blank.trials.values,timeID='visontimes').values
        elif align_type == 'aud':
            incongruent_times = MS.sel(visazimuths=myVazi,contrast=myContrast,audazimuths=myAazi,SPL=mySPL,trials=blank.trials.values,timeID='audontimes').values

        trig_times_matrix[c_idx,:] = incongruent_times[:n_trials]
        #vis weights
        p_idx_v=np.where(params['visAzimuths']==myVazi)[0]
        indicator_matrix[c_idx,p_idx_v]=1
        #aud weights
        ixA=np.where(params['audAzimuths']==myAazi)[0]
        p_idx_a = params['visAzimuths'].size + ixA
        indicator_matrix[c_idx,p_idx_a]=1
        c_idx += 1

    # write details of indicator matrix

    indicator_matrix_={
        "array": indicator_matrix,
        "x_dim": params, # concantenated visual and aud azimuth
        "y_dim": conditions, # concantenated array of condition types 
        "y_vis_azimuths":vis_azis,
        "y_aud_azimuths":aud_azis,
    }

    return trig_times_matrix,indicator_matrix_

def get_all_indicator_matrices(indicator_matrix_,fit_type='Linear',alpha=None): 
    # 
    indicator_matrix = indicator_matrix_['array']
    params = indicator_matrix_['x_dim']
    conds = indicator_matrix_['y_dim']
    trialtypes_sizes=np.zeros(5).astype('int')
    for i,c in enumerate(conds.keys()):
        trialtypes_sizes[i]=conds[c] 

    visAzimuths=params['visAzimuths']
    audAzimuths=params['audAzimuths']
    visual_indicator_matrix=indicator_matrix[:,:trialtypes_sizes[1]]
    auditory_indicator_matrix=indicator_matrix[:,trialtypes_sizes[1]:]
    # the auditory center only model 
    IM_audC=indicator_matrix.copy()
    audC_idx=visAzimuths.size+np.where(audAzimuths==0)[0][0] 
    IM_audC[np.sum(trialtypes_sizes[:2]):,audC_idx]=1 #mk

    # baseline firing rate
    myones=np.ones(IM_audC.shape[0]) # the firing rate thing -- I am not sure this is needed if we baseline subtract? 
    IM_audC=np.concatenate((IM_audC,myones[:,np.newaxis]),axis=1)

    # columnt indices for the various matrices
    bl_idx=IM_audC.shape[1]-1
    Vs=np.append(np.arange(visAzimuths.size),bl_idx)
    Ac=np.append(audC_idx,bl_idx)
    VsAc=np.sort(np.append(Vs,audC_idx))

    start_end_idx = np.cumsum(trialtypes_sizes)
    start_end_idx = np.append(np.array([0]),start_end_idx)
    # introducing the multisensory term for the congruent trials only
    IM_nl = IM_audC.copy()
    congruent_idx = np.where(['multi_congruent' in k for k in conds.keys()])[0]
    for looping_idx in range(conds['multi_congruent']): 
        msterm=np.zeros(IM_audC.shape[0])
        msterm[start_end_idx[congruent_idx]+looping_idx] = 1
        IM_nl=np.concatenate((IM_nl,msterm[:,np.newaxis]),axis=1)
        


    # could write an xarray of all matrices? well they are now
    IM={'V_spatial_A_spatial': IM_audC, #y
        'V_spatial':IM_audC[:,Vs], #y
        'A_spatial':IM_audC[:,visAzimuths.size:],  #y
        'A_center': IM_audC[:,Ac], #y
        'V_spatial_A_center': IM_audC[:,VsAc], #y
        'baseline':IM_audC[:,bl_idx][:,np.newaxis], #y
       'nonlinear': IM_nl,
        } 

    if 'Ridge' in fit_type: 
        for keys in IM.keys():
            IM[keys] = conc_ridgeMatrix(IM[keys],alpha)
    
    return IM

def conc_ridgeMatrix(design_matrix,myLambda,axis=1):
    # axis=1 addig identity to columns
    if axis==1: 
        myI=np.identity(design_matrix.shape[1])*myLambda
        outM=np.concatenate((design_matrix,myI),axis=0)
        
    return outM

def divide_to_crossval_sets(array,train_size=.5,subselection=None,random_state=0):
    """
    divide array to cross-validation sets. 
    :param array: single array either with timepoints or array of indices
    :param train_size: float between 0 and 1. Fraction of training set
    :subselection: select this many no. of trials from array
    :random state: int if you want to shuffle, None if you don't 
    """
    if random_state!=None:
        np.random.seed(random_state)
        np.random.shuffle(array)

    if subselection!=None:
        arrays=array[:subselection]
    
    n_trials=len(array)
    n_train=int(np.ceil(n_trials*train_size))
    n_sets=[0,n_train,n_trials]
    
    sets={}
    for sidx in range(len(n_sets)-1):
        sets['set_%.0d' % sidx]=array[n_sets[sidx]:n_sets[sidx+1]]
    
    
    return Bunch(sets)

def ols_coeff(X, Y):
    xTx = np.einsum('...ji, ...jk -> ...ik', X, X)
    # inv_cov = np.linalg.inv(xTx)
    inv_cov = np.tile(np.linalg.inv(xTx[0]), (xTx.shape[0], 1, 1))
    term = np.einsum('...ij, ...kj', inv_cov, X)
    betas = np.einsum('...ij, ...jk', term, Y)
    pred = np.einsum('...ij, ...jk -> ...ik', X, betas)
    return betas, pred

def fit_additive_ols(response_matrix_train,indicator_matrix,fit_type='Linear'):
    response_timepoints=response_matrix_train.shape[2]
    goodclusno=response_matrix_train.shape[0]
    # weights=np.zeros((goodclusno,response_timepoints,indicator_matrix.shape[1]))
    # prediction_matrix=np.zeros((goodclusno,indicator_matrix.shape[0],response_timepoints))
    out = []
    for nrn in range(goodclusno):
        for t in range(response_timepoints):
            r = response_matrix_train[nrn, :, t]
            
            if 'Ridge' in fit_type:
                r=np.concatenate((r,np.zeros(indicator_matrix.shape[1])))            
            
            out.append(r)

    yy = np.stack(out)[:, :, None]
    xx = np.tile(indicator_matrix, (len(out), 1, 1))
    _, pred = ols_coeff(xx, yy)
    pred = pred.reshape(goodclusno, response_timepoints, -1)
    pred = pred.transpose(0, 2, 1)  # swap the last two axis
    return pred

def fit_additive_model(response_matrix_train,indicator_matrix,fit_type='Linear'): 
    # response_matrix train should only contain timepoints that are to be fitted 
    response_timepoints=response_matrix_train.shape[2]
    goodclusno=response_matrix_train.shape[0]
    weights=np.zeros((goodclusno,response_timepoints,indicator_matrix.shape[1]))
    prediction_matrix=np.zeros((goodclusno,indicator_matrix.shape[0],response_timepoints))
    for nrn in range(goodclusno):
        for t in range(response_timepoints):
            r=response_matrix_train[nrn,:,t]
            if 'Ridge' in fit_type: 
                # if we fit the ridge model (i.e. added the identity at the bottom of the indicator, then need to shove 0s to r)
                # if we added the right matrix then the 2nd dimentsion of the design matrix ought to be exactly the size of 
                # the 0s I need to add to the predicted values 

                r=np.concatenate((r,np.zeros(indicator_matrix.shape[1])))

            
            reg=LinearRegression().fit(indicator_matrix,r)
            #weights[nrn,t,:]=reg.coef_
            
            prediction_matrix[nrn,:,t]=reg.predict(indicator_matrix)

    if 'Ridge' in fit_type:
        # basically we will not be needing the predicitons for the regulaisers 
        predicted=predicted[:,:response_matrix_train.shape[1],:]

    return prediction_matrix

def get_VE(actual,predicted):
    allVE=np.zeros(actual.shape[0])

    for nrn in range(actual.shape[0]):
        all_actual=np.ravel(actual[nrn,:,:])
        all_predicted=np.ravel(predicted[nrn,:,:])
        myVE = 1-(np.var(all_actual-all_predicted)/np.var(all_actual))
        allVE[nrn]=myVE

    return (allVE)

class AV_model():
    def __init__(self,regulariser):
        self.regulariser = regulariser

    def load_data(self,**rec_info):
        # load the event data by default 
        ephys_dict =  {'spikes': ['times', 'clusters'],'clusters':'all'}
        other_ = {'events': {'_av_trials': 'table'}}

        recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=other_,**rec_info)
        if recordings.shape[0] == 1:            
            recordings =  recordings.iloc[0]
        else:
            print('recordings are ambiguously defined. Please recall.')
        events,self.spikes = recordings.events._av_trials,recordings.probe.spikes

        
        self.b,self.v,self.a,self.ms = postactive(events)
        # give dattype kwarg or call default

        self.dattype_kwargs = {
            'blank_reps':0,
            'myContrast':np.max(self.v.contrast.values),
            'mySPL':np.max(self.a.SPL.values),
            'n_trials':30,
            'align_type':'vis'
        }

        self.bin_kwargs = {
            'tscale':[None],
            'pre_time':0,'post_time': .2, 
            'bin_size':0.005, 'smoothing':0.02                            
        }
        
    def format_spike_data(self,subselect_neurons=None):            
            t_trig,indicator_matrix=sort_trialtypes_to_conditions(self.b,self.v,self.a,self.ms,**self.dattype_kwargs)

            if subselect_neurons:
                self.clusIDs = np.array(subselect_neurons)
            else:
                self.clusIDs = np.unique(self.spikes.clusters)

            r = [su.get_binned_rasters(self.spikes.times,self.spikes.clusters,self.clusIDs,t_trig[i,1:],**self.bin_kwargs) for i in range(t_trig.shape[0])]

            rasters = np.zeros((len(r),r[0]['rasters'].shape[0],r[0]['rasters'].shape[1],r[0]['rasters'].shape[2]))
            for i in range(len(r)):
                rasters[i,:,:,:] = r[i]['rasters']

            self.rasters = rasters
            self.rasters_tscale= r[0]['tscale']
            self.rasters_cscale=r[0]['cscale']
            self.additive_indicator_matrix = indicator_matrix # maybe add aud and vis azimuth to each

    
    def get_psths(self,train_idx=None,test_idx=None,random_state=0):

        # can also input your own indices
        if train_idx is None: 
            self.random_state = random_state 
            cval_idx = divide_to_crossval_sets(np.arange((self.rasters.shape[1])),train_size=0.5,subselection=30,random_state=self.random_state)
            train_idx= cval_idx['set_0']
            test_idx = cval_idx['set_1']
            

        # write a dictinary instead # get nanmean as some values will become nans in the pipeline framework

        self.psth_train = np.nanmean(self.rasters[:,train_idx,:,:],axis=1).transpose(1,0,2)
        self.std_train = np.nanstd(self.rasters[:,train_idx,:,:],axis=1).transpose(1,0,2)
        self.sem_train = self.std_train/np.sqrt(train_idx.size)

        if test_idx is not None: 
            
            self.psth_test = np.nanmean(self.rasters[:,test_idx,:,:],axis=1).transpose(1,0,2)
            self.std_test = np.nanstd(self.rasters[:,test_idx,:,:],axis=1).transpose(1,0,2)
            self.sem_test = self.std_test/np.sqrt(test_idx.size) 

            self.score_retest = get_VE(self.psth_test,self.psth_train) 

            return self.score_retest
    
    def fit_predict(self,ind_matrix=None):
        if ind_matrix is None:
            ind_matrix_ = self.additive_indicator_matrix
        else: 
            ind_matrix_ = ind_matrix

        #self.predictions = AVm.fit_additive_model(self.psth_train,ind_matrix_,self.regulariser)
        self.predictions = fit_additive_ols(self.psth_train, ind_matrix_,fit_type=self.regulariser)

        if 'Ridge' in self.regulariser:
            self.predictions=self.predictions[:,:self.psth_test.shape[1],:]
        
        self.score_model = get_VE(self.psth_test,self.predictions)

        return self.score_model

    def fit_list_of_models(self,IMs,random_state):
        # this ought to take a dictionary of models
        score_retest = self.get_psths(random_state=random_state) 
        scores_model = [self.fit_predict(ind_matrix=IMs[m]) for m in IMs.keys()]
        scores = {'full': score_retest}
        for score,model in zip(scores_model,IMs.keys()):
            scores[model]=score
        
        return scores

    def plot_psths(self,nrn,trainON=True,predON=False,testON=False,ax=None,fig=None,plotted_vis_azimuth=None,plotted_aud_azimuth=None,traincolor='grey',predcolor='red',testcolor='grey',predlw=10):
        
        if plotted_aud_azimuth is None: 
            plotted_aud_azimuth = np.unique(self.additive_indicator_matrix["y_aud_azimuths"])
        if plotted_vis_azimuth is None: 
            plotted_vis_azimuth = np.unique(self.additive_indicator_matrix["y_vis_azimuths"])
        
        vazi,aazi=np.meshgrid(plotted_vis_azimuth,plotted_aud_azimuth)
        
        #just in case I need to give ax externally...
        if ax is None: 
            plt.rcParams.update({'font.family':'Verdana'})
            plt.rcParams.update({'font.size':14})
            plt.rcParams['figure.dpi'] = 300

            fig,ax=plt.subplots(plotted_vis_azimuth.size,plotted_aud_azimuth.size,figsize=(7,7),sharey=True)
            fig.patch.set_facecolor('xkcd:white')


        for i,m in enumerate(vazi):
                for j,_ in enumerate(m):
                    v = vazi[i,j]
                    a = aazi[i,j]
                    idx = np.where((self.additive_indicator_matrix["y_vis_azimuths"]==v) & 
                                    (self.additive_indicator_matrix["y_aud_azimuths"]==a))[0]
                
                    

                    if idx.size==1:
                        myax = ax[plotted_aud_azimuth.size-1-i,j]
                        #myax.clear()
                        # most certainly plot the training set
                        if trainON:
                            my_plotter.plot_sempsth(self.psth_train[nrn,idx[0],:],self.sem_train[nrn,idx[0],:],self.rasters_tscale,
                                        myax,errbar_kwargs={'color': traincolor, 'alpha': 0.4})

                        # can also choose to plot the test set and the pediction

                        if testON:
                            my_plotter.plot_sempsth(self.psth_test[nrn,idx[0],:],self.sem_test[nrn,idx[0],:],self.rasters_tscale,
                                    myax,errbar_kwargs={'color': testcolor, 'alpha': 0.4})
                                    
                        if predON:
                            myax.plot(self.rasters_tscale,self.predictions[nrn,idx[0],:],color=predcolor,lw=predlw,alpha=.95)
                             
                        my_plotter.off_axes(myax)
                        

                    my_plotter.off_axes(ax[i,j])

        for i in range(plotted_aud_azimuth.size):
            ax[plotted_aud_azimuth.size-1-i,0].set_ylabel('%s' % int(plotted_aud_azimuth[i]))
            ax[plotted_aud_azimuth.size-1,i].set_xlabel('%s' % int(plotted_vis_azimuth[i]))

  #      if fig==None:
        fig.text(.44,0.05,'visual azimuth',weight = 'bold')
        fig.text(0.05,0.425,'auditory azimuth',weight = 'bold',rotation=90)

def save_psth_model_results(recordings,probe,model,IMs,scores,scores_to_clusters=True):
        saveroot = r'C:\Users\Flora\Documents\Processed data\Audiovisual'
        mname = recordings.Subject
        date = recordings.expDate
        expNum = recordings.expNum
        savepath = '%s\%s\%s\%s\imec%.0f\AV_psth_model' % (saveroot,mname,date,expNum,probe)        

        savepath = '%s\%s' % (savepath,model.regulariser)

        if not os.path.exists(savepath):
                os.makedirs(savepath)

        f=open(os.path.join(savepath,"scores.pcl"),'wb')
        pickle.dump(scores,f)
        f.close()

        f=open(os.path.join(savepath,"indmatrices.pcl"),'wb')
        pickle.dump(IMs,f)
        f.close()

        f=open(os.path.join(savepath,"model.pcl"),'wb')
        pickle.dump(model,f)
        f.close()

        # also save things that are cluster based and will possibly be used for subseqeunt analysis
        if scores_to_clusters: 
            model_names = list(scores[0].keys())
            scores_all_reps=dict.fromkeys(model_names,[])
            for m in model_names:
                meanscore = np.array([scores[i][m]  for i in range(len(scores))])
                scores_all_reps[m]  = meanscore 

                np.save(r'%s\clusters.ve_%s.npy' % (savepath,m),meanscore)
                
            meanVE  = np.array([scores_all_reps[m].mean(axis=0) for m in model_names])
            WinnerModel = np.array([model_names[np.argmax(meanVE,axis=0)[i]] for i in range(meanVE.shape[1])])
            np.save(r'%s\clusters.winner_took_it_all.npy' % (savepath),WinnerModel)
            
def load_psth_model_results(savepath):
    f=open(os.path.join(savepath,"indmatrices.pcl"),'rb')
    IMs=pickle.load(f)
    f.close()

    f=open(os.path.join(savepath,"scores.pcl"),'rb')
    scores=pickle.load(f)
    f.close()


    f=open(os.path.join(savepath,"model.pcl"),'rb')
    model=pickle.load(f)
    f.close()

    return model,IMs,scores

def get_winner_from_scores(scores):
    model_names = list(scores[0].keys())
    scores_all_reps=dict.fromkeys(model_names,[])
    for m in model_names:
        scores_all_reps[m]  = np.array([scores[i][m]  for i in range(len(scores))])
    meanVE  = np.array([scores_all_reps[m].mean(axis=0) for m in model_names])
    WinnerModel = np.array([model_names[np.argmax(meanVE,axis=0)[i]] for i in range(meanVE.shape[1])])

    return WinnerModel

def plot_Winning_cellpos(WinnerModel,xpos,ypos,ax0=None,ax1=None):

    """
    plot cell position along 4-shank probes and color according to the WinnerModel
    """

    if ax0==None:
        _,ax = plt.subplots(2,1,figsize=(7,10),gridspec_kw={'height_ratios':[1,5]})
        ax0=ax[0]
        ax1=ax[1]

    def plot_selected(x,y,ix,myc,mys,mya,ax):
        ax.scatter(x[ix],y[ix], c=myc,s=mys,alpha=mya)

    myMorder=['baseline','full','V_spatial_A_spatial','V_spatial','A_spatial','A_center','V_spatial_A_center']
    colors=['grey','black','green','blue','magenta','plum','lightgreen']

    rand_num = np.random.normal(0,1.5,xpos.size)
    M,Mcounts=np.unique(WinnerModel,return_counts=True)
    
    ax1.vlines(20,0,2880*2,'k',alpha=.1)
    ax1.vlines(220,0,2880*2,'k',alpha=.1)
    ax1.vlines(420,0,2880*2,'k',alpha=.1)
    ax1.vlines(620,0,2880*2,'k',alpha=.1)

    for mix,model in enumerate(myMorder):
        ix=np.where((WinnerModel==model))[0]
        myc=colors[mix]
        plot_selected(xpos+rand_num,ypos,ix,myc,2,1,ax1)


        count_idx =np.where(M==model)[0]
        if count_idx>0:
            ax0.bar(mix,Mcounts[count_idx],color=myc)
            ax0.text(mix-.3,Mcounts[count_idx]+10,model,rotation=45,fontsize=6)
    
    my_plotter.off_axes(ax0)
    my_plotter.off_topspines(ax1)

def get_meanscore(scores):
    model_names = list(scores[0].keys())
    scores_all_reps=dict.fromkeys(model_names,[])
    for m in model_names:
        scores_all_reps[m]  = np.array([scores[i][m]  for i in range(len(scores))])
    meanVE  = np.array([scores_all_reps[m].mean(axis=0) for m in model_names])
    WinnerModel = np.array([model_names[np.argmax(meanVE,axis=0)[i]] for i in range(meanVE.shape[1])])
    return meanVE, WinnerModel


def default_fitting(rec_info,repeats=1):
    """
    default fitter that fits the 8 models Linearly (no regularisation)
    (full, A,V,A&V spatials, Acenter,AVcenter,nonlinear,baseline)

    Parameters:
    ----------
    rec_info: pd.Series that contain indentification for loading using the av_pipeline
    repeats: how many times to repat the fitting procedure

    Returns: 
    --------
        :pd.Dataframe containing mean variance explained and the winner model for each cluster
    """
    
    
    av_model = AV_model('Linear')
    av_model.load_data(**rec_info)
    av_model.format_spike_data()

    IMs = get_all_indicator_matrices(
        av_model.additive_indicator_matrix,
        fit_type=av_model.regulariser,
        alpha=.0001
        )

    scores = [av_model.fit_list_of_models(IMs, myrepeat) for myrepeat in range(repeats)]

    model_names = list(scores[0].keys())
    scores_all_reps=dict.fromkeys(model_names,[])
    for m in model_names:
        meanscore = np.array([scores[i][m]  for i in range(len(scores))])
        scores_all_reps[m]  = meanscore 
        
    meanVE  = np.array([scores_all_reps[m].mean(axis=0) for m in model_names])
    meanVE = pd.DataFrame(meanVE.T,columns=model_names)
    meanVE['winner_model'] = meanVE.idxmax(axis=1)
    meanVE = meanVE.set_index(av_model.clusIDs,drop = True)

    return meanVE