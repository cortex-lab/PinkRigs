# PinkRigs
Shared code for running experiments and processing data on the pink rigs

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
### Analysis
## histology 
- '\Analysis\+hist\Save_images_run_brainreg.ipynb' automatically runs brainreg. To run, open jupyter notebook in evironment where you installed brainreg. 
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
`conda env export > environment.yml`


