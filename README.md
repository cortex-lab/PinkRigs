# PinkRigs
Shared code for running experiments and processing data on the pink rigs

## Notes on the CSVs

## Python and Matlab
### Notes on who does what

In this pipeline, Python is mainly used for spike sorting, IBL formatting, histology, and video processing (FaceMap and DLC). Matlab is mainly used for data management (CSVs, etc.) and data alignment and formatting. Both are used for analysis.

### Standard Matlab function input format

Most scripts in Matlab use a standard input format.

### Using python scripts
#### Install for the first time
1. After cloning the environment,open anaconda prompt. 
2. Run `conda env create -f environment.yml`
3. Then activate the environment by `conda activate PinkRigs`

#### update after git pull
1. In Anaconda prompt, activate the environment by `conda activate PinkRigs`
2. Run `conda env update --file environment.yml --prune`

#### Dev
If you added new packages to the environment, overwrite the current `environment.yml` by running the following before you git add/commit: 
`conda env export -f environment.yml`

#### installing FFmpeg on Windows 
the PinkRig environment is dependent `FFmpeg` which needs to be installed manually and added to system varaibles in Windows. [Here is an installation guide that worked.](https://phoenixnap.com/kb/ffmpeg-windows)


## Regular automated scripts 
The pipeline runs automatically, mostly at night. It deals with the data that is on the experimental computers, to produce a standardized output for each experiment.

Here is a list of the automated tasks, and when they start:
### On timeline computer
- 8pm: 
  - Push daily camera data to server
  - Run FaceMap 
### On ephys computer
- 8pm: 
  - Push daily mic data to server
  - Extract the sync and compress all ephys data (older than 1h)
  - Copy compressed ephys data and associated files
  - Run FaceMap
### On kilo computer
- 22pm:
    - Send update on training via email
- 22pm/1am    
    - Run pyKilosort ont he queue
    - Convert to IBL format
    - If after 2am, run the alignment and extract data
All of this is saved locally in a log file.

We use Windows task scheduler to set up the tasks: 
1. Select `Create a basic task...`
2. Select the relevant timings and when prompted to select the task to schedule, select, the relevant batch file:
`\Github\PinkRigs\Admin\nightManager.bat`

There are also other processed that are semi-automated:
### Histology 
- `\Analysis\\+hist\Save_images_run_brainreg.ipynb` automatically runs brainreg. To run, open jupyter notebook in evironment where you installed brainreg. 

## Manual scripts 
### Running sorting manually 

### Running alignment manually 

### 'Manual curation' of experiments

1. If you record and experiment which, for whatever reason, is not worth keeping, please add "IgnoreExperiment.txt" to the experiment path on the server (see \\zinu.cortexlab.net\Subjects\CB020\2021-11-25\1) for an example. This will mean the experiment isn't added to the .csv and so you won't have to deal with inevitable errors etc.
2. Alternatively, if an experiment is worth keeping, but you have checked the errors and you are satisfied that nothing can be done (e.g. you want to keep the behavior even though the flipper wasn't working), please add "AllErrorsValidated.txt" to the experiment folder. (see \\zinu.cortexlab.net\Subjects\FT025\2021-10-06\1 for an example)
In both cases, please write a few words to explain in the text file. In this way we can continue to keep our csv's tidy, with an accurate reflection of the current state of processing.

## Running analysis (Python)

### Querying experiments

You can query `csv_queryExp.load_data` module, e.g.:
```
from Admin.csv_queryExp import queryCSV


exp = queryCSV(
  subject='AV043',
  expDate='2024-03-14:2024-03-24', 
  expDef = 'multiSpaceWorld_checker_training',
  )
```

### Loading the data
You can also direcrly query and then load the ONE folder content in one line using 'load_data'. To specify the ONE folder content to load, you need to give a nested dictionary to the 'data_name_dict' argument of the 'load_data'. The nesting follows the ONE data structure '{collection:{'object':'attribute'}}'. For example: 
#### Events data
```
from Admin.csv_queryExp import load_data

# define parameters of your query
exp_kwargs = {
    'subject': ['AV043','AV044'],
    'expDate': '2024-03-14:2024-03-24',
    }

# define the ONE data to load
data_name_dict = { 'events': {'_av_trials': 'table'})
recordings = load_data(data_name_dict=data_name_dict,**exp_kwargs)
```
#### Spikes data
```
ephys_dict = {'spikes':'all','clusters':'all'}
# both probes 
data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict} 
recordings = load_data(data_name_dict=data_name_dict,**exp_kwargs)
```
#### Camera data
```
cameras = ['frontCam','sideCam','eyeCam']
data_name_dict = {cam:{'camera':['times','ROIMotionEnergy']} for cam in cameras}
recordings = load_data(data_name_dict=data_name_dict,**exp_kwargs)
```

Alternatively you can also first query the data using `csv_queryExp`, subset your DataFrame as you wish, and load the ONE object only on your subset using 'load_data'.

```
load_data(recordings=exp.iloc[0:1], data_name_dict = {'events':{'_av_trials':'all'}})

```
You can also try it out yourself [here](https://github.com/cortex-lab/PinkRigs/blob/main/Analysis/data_loading_examples.ipynb). 

## Miscellaneous yet useful functions
### When planning an experiment
#### `plt.recLocation`
#### `+imro` package

### When doing an experiment
#### `changeMouseNameAndExpNum`

### Checking the data
#### Behaviour

#### Spikes

#### Videos

#### Mic

### Debugging
#### `plt.functionGraph`
#### `checkOrChangePinkRigsFunc`
Use this function to change the name of a function in the whole repo.
