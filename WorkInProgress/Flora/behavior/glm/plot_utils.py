
# for computations
import numpy as np
import pandas as pd 

from sklearn.linear_model import LogisticRegression

# for plot handling 
import matplotlib.pyplot as plt
from io import StringIO
import pyperclip

def get_region_colors():
    """
    HEX codes of the different brain regions I often use
    """
    return {
        'SC': '#D392C0',
        'Frontal': '#3CA37B',
        'Vis':'#278F8F',
        'Parietal': '#28A3B7',
        'Lateral': '#00BCD1',
    }

def off_axes(ax,which='all'):
    """
    plot to turn off certain axes
    Parameters: 
        ax: the matplotlib ax object 
        which: str
            options: 'all','top','exceptx','excepty'
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    offx, offy = False, False
    if which=='all': offx = True 
    if which=='excepty': offx = True 

    if which=='all': offy = True
    if which=='exceptx': offy = True 

    if offx: 
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_xlabel('')

    if offy: 
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel('')        


def copy_svg_to_clipboard(fig):
    """
    function that can directly grab a matplotlib figure and copy to clipboard as an SVG.

    """

    # Step 2: Save the figure as an SVG string
    svg_io = StringIO()
    fig.savefig(svg_io, format='svg')
    #plt.close(fig)  # Close the plot so it doesn't display twice

    # Step 3: Get the SVG content as a string
    svg_code = svg_io.getvalue()

    # Step 4: Copy the SVG string to the clipboard
    pyperclip.copy(svg_code)

    print("SVG content copied to clipboard!")



def plot_psychometric(trials,gamma=1, weights = None,
                     yscale='log',ax=None, plot_curve = True, refit = True, 
                     dataplotkwargs={'marker':'o','ls':''},
                     predpointkwargs ={'marker':'*','ls':''},
                     predplotkwargs={'ls':'-'}):
    
    """
    plot the model prediction for this specific model
    if the model has neural components we 0 those out

    refit: bool
        whether to refit the AV model -- in this case we use the refitted model weights as predictions

    """
    if ax is None:
        _,ax = plt.subplots(1,1,figsize=(8,8))
    

    assert np.isin(['visR','audR','visL','audL','choice'],trials.columns).all(),(
        'the trials are inputted in an unexpected format.'
    )
    # handling gamma -- for now we use common gamma
    for v in ['visR','visL']:
        trials[f'{v}_gamma'] = trials[v] ** gamma

    # get commonly used variables
    visDiff = trials.visR_gamma - trials.visL_gamma
    audDiff = trials.audR - trials.audL
    choices = trials.choice
    Vs = np.unique(visDiff)
    As = np.unique(audDiff)

    # determine the colors we will use in the plot using coolwarm as colormap
    colors=plt.cm.coolwarm(np.linspace(0,1,As.size))


    # refit the model
    m = LogisticRegression()
    fitted_features = ['visR_gamma','audR','visL_gamma','audL']
    m.fit(trials[fitted_features],trials['choice'])

    # else we take the model weights but I think that is not always straightforward (e.g. with the neurons - but can be more useful in other conditions)


    # plot the actual data
    Vmesh,Amesh =np.meshgrid(Vs,As)

    for v,a,mycolor in zip(Vmesh,Amesh,colors):
        x = v
        y  = np.array([np.mean(choices[(visDiff==vi) & (audDiff==ai)]) for vi,ai in zip(v,a)])
        if yscale=='log':
            y =np.log(y/(1-y))        

        ax.plot(x,y,color=mycolor,**dataplotkwargs)


    # plot the precition only on the data  points, I don't really use it. 

        # if weights is not None: 
        #     logOdds = np.dot(trials,weights)
        #     pR = np.exp(logOdds) / (1 + np.exp(logOdds))
        #     y_pred = np.array([np.mean(pR[(visDiff==vi) & (audDiff==ai)]) for vi,ai in zip(v,a)])
        # else:
        #     y_pred = np.empty(x.size)*np.nan # nan if there is no prediction 

        # if yscale=='log':
        #     y_pred = np.log(y_pred/(1- y_pred))
       
        # ax.plot(x,y_pred,color=mycolor,**predpointkwargs)



    #plotting the prediciton psycometric as a line

    if plot_curve:
        # get dense points for vis
        nPredPoints = 600  
        Vmodel = np.linspace(-1,1,nPredPoints) 
        # gamma transform
        visDiff_pseudo = np.sign(Vmodel)*np.abs(Vmodel)**gamma 
        x_  = visDiff_pseudo
        # get samples for auditory
        audDiff_pseudo = np.ones(nPredPoints)*As[:,np.newaxis]

        for a,mycolor in zip(audDiff_pseudo,colors):
            # re-create matrix with what we are predicting
            pseudo_trials = pd.DataFrame()
            pseudo_trials['visL_gamma'] = np.abs(visDiff_pseudo) * (visDiff_pseudo<0)
            pseudo_trials['visR_gamma'] = np.abs(visDiff_pseudo) * (visDiff_pseudo>0)
            pseudo_trials['audL'] = np.abs(a) * (a<0)
            pseudo_trials['audR'] = np.abs(a) * (a>0)

            assert np.isin(pseudo_trials.columns, fitted_features).all(),(
                'the pseudo trial features differ from the model..'
            )
            
            # predict with the model
            # if yscale=='log':
            #     y_ = np.exp(y_) / (1 + np.exp(y_))
            y_ = m.predict_proba(pseudo_trials[fitted_features])[:,1] 
            if yscale=='log':
                y_ =np.log(y_/(1-y_))        


            ax.plot(x_,y_,color=mycolor,**predplotkwargs)


    # plot the guider lines  
    if yscale=='log':
        ax.axhline(0,color='k',ls='--')
    else:
        ax.axhline(.5,color='k',ls='--')

    ax.axvline(0,color='k',ls='--')

