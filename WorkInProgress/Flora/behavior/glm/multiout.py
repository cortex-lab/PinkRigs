# %%
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

trials = pd.read_csv(r'D:\LogRegression\opto\uni_all_nogo\formatted\AV046_right_10mW.csv')
#%%

trials = trials[(trials.choice==0) | (trials.choice==1) | (trials.choice==-1)]

# balanced subsampling attempts
# opto_0 = trials[trials['choice'] == 0]
# opto_1 = trials[trials['choice'] == 1]
# opto_2 = trials[trials['choice'] == -1]

# min_size = min(len(opto_0), len(opto_1), len(opto_2))

# # Randomly sample from both groups to make the sizes equal
# opto_0_sampled = opto_0.sample(n=min_size, random_state=0)
# opto_1_sampled = opto_1.sample(n=min_size, random_state=0)
# opto_2_sampled = opto_2.sample(n=min_size, random_state=0)

# # Combine the sampled DataFrames
# balanced_trials = pd.concat([opto_0_sampled, opto_1_sampled, opto_2_sampled])

# trials  = balanced_trials

stim_predictors = ['visR','visL','audR','audL']
opto_slope_predictors = ['visR_opto','visL_opto','audR_opto','audL_opto']
opto_intercept_predictor = ['opto']
all_predictors =  stim_predictors + opto_slope_predictors + opto_intercept_predictor


#%%


X = trials[all_predictors]
y = trials['choice']
stratifyIDs = trials['trialtype_id']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1,shuffle=True,stratify=stratifyIDs)

m = LogisticRegression()
m.fit(X_train,y_train)
# %%
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test,m.predict(X_test))
plt.imshow(cm)

# %%
# might be a bit unbalanced


plt.matshow(m.coef_,cmap='coolwarm',
            vmin=-3,vmax=3)

pred_names = list(X_train.keys())
plt.xticks(ticks = range(len(pred_names)),
           labels = pred_names, 
            rotation=45)
# %%
