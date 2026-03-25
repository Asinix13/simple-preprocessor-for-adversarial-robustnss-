All code in here is adapted from: https://github.com/SPIN-UMass/MeanSparse except my_utils.py.

To change the hyperparameters for any file, you need to go into the file the hyperparameter section is marked there.
And all hyperparameters should be described.

For the AA_test.py and for the noise_test.py some hyperparameters are changed with an argument parser.
Those are listed in the beginning of those files.

The SotA model used needs to be downloaded from https://github.com/SPIN-UMass/MeanSparse. 
It is the Sparsified_WRN_94_16_CIFAR model, which needs to be put into the models_WS folder.

To run any file use:
python3 file_name.py

To test against Auto Attack use AA_test.py
To test against different noises use noise_test.py.

These to above files will creat a logger file under results/name_without_test/Restults.txt which will include the accuracys.

To test against all adversarial attacks except C&W and Auto Attack use mult_adv.py.
The results will be in the folder mult_adv/time/ as acc.xlsx.

To test against C&W attacks use CW_adv.py.
The results will be in the folder CW_adv/time/ as acc.xlsx, L2.xlsx distance file is decreaped for models that use the normalized data.