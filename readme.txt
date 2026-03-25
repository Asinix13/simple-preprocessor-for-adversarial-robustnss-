Create and activate the conda env for this thesis with:
conda env create -f environment.yml
conda activate ma_env

All test concerning the Vanilla model are in the Standard folder;
the ones for the SotA WRN models in the SotA folder.
Both have their own readme and both need the conda env ma_env.

If code is used from open source githubs it will be mentioned directly in the code and in the respective readme files:
For BPDA we use the code from: https://github.com/Annonymous-repos/attacks-in-pytorch/blob/master/attacks/BPDA.py

All hyperparameters are listed in the paper and for the SotA experiments there are config files for each run.
