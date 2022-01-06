# PinkRigs
Shared code for running experiments and processing data on the pink rigs


## For using python scripts
### Install for the first time
1. After cloning the environment,open anaconda prompt. 
2. Run `conda env create -f environment.yml`
3. Then activate the environment by `conda activate PinkRigs`

### update after git pull
1. In Anaconda prompt, activate the environment by `conda activate PinkRigs`
2. Run `conda env update --file environment.yml --prune`

### Dev
If you have added new packages to the environment,overwrite the current environment.yml by running the following before you add/commit in the environment: 
`conda env export > environment.yml`


