# PinkRigs
Shared code for running experiments and processing data on the pink rigs
## Standard variable names (format)
- "subject": Name of the subject (string)
- "expDate": Date of the experiment as yyyy-mm-dd (string)
- "expNum": Experiment number on a given day (string)


## Regular automated scripts 
### timeline
- 8pm: push daily camera data to server and deletion of all *copied* data that is >2 days old. ('Zelda-Time\delete_expData.bat')
- 8pm: facemap 
### ephys
- 8pm: push daily ephys data to server and deletion of all *copied* data that is >2 days old. ('Zelda-Ephys\delete_expData.bat') 
### data manager
- 6.30pm inform users of training ('Data-Management\check_training.bat')
- 9pm: check for new experiments, new ephys data to sort, initialise kilosort ('Data-Management\check_exps.bat') 

We use Windows task scheduler to regulate this. 
To set up regular deletion of camera data on timeline/ephys computers on Windows: 
1. Select `Create a basic task...`
2. Select the relevant timings and when prompted to select the task to schedule, select, the relevant batch file e.g.
`\Github\PinkRigs\Zelda-Time\delete_expData.bat`

## Analysis
### histology 
- `\Analysis\\+hist\Save_images_run_brainreg.ipynb` automatically runs brainreg. To run, open jupyter notebook in evironment where you installed brainreg. 
### Preprocessing
- `\Analysis\+preproc\main` will run the alignment for the ephys, block, videos, and microphone, to timeline, and preprocess the data. It can take a list of experiments as an input, or will go through all experiments of active mice.
- `\Analysis\+preproc\+align\main` will compute the `alignment.mat` file for a list of experiments.
### Spikesorting
- `\Analysis\+kilo\main` will run Kilosort2, either on a list of recordings (given as an input) or on the waiting list. 

## Using python scripts
### Install for the first time
1. After cloning the environment,open anaconda prompt. 
2. Run `conda env create -f environment.yml`
3. Then activate the environment by `conda activate PinkRigs`

### update after git pull
1. In Anaconda prompt, activate the environment by `conda activate PinkRigs`
2. Run `conda env update --file environment.yml --prune`

### Dev
If you added new packages to the environment, overwrite the current `environment.yml` by running the following before you git add/commit: 
`conda env export -f environment.yml`

## 'Manual curation' of experiments

1. If you record and experiment which, for whatever reason, is not worth keeping, please add "IgnoreExperiment.txt" to the experiment path on the server (see \\zinu.cortexlab.net\Subjects\CB020\2021-11-25\1) for an example. This will mean the experiment isn't added to the .csv and so you won't have to deal with inevitable errors etc.
2. Alternatively, if an experiment is worth keeping, but you have checked the errors and you are satisfied that nothing can be done (e.g. you want to keep the behavior even though the flipper wasn't working), please add "AllErrorsValidated.txt" to the experiment folder. (see \\zinu.cortexlab.net\Subjects\FT025\2021-10-06\1 for an example)
In both cases, please write a few words to explain in the text file. In this way we can continue to keep our csv's tidy, with an accurate reflection of the current state of processing.
