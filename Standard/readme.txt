To change the hyperparameters for any file, you need to go into the file the hyperparameter section is marked there.
And all hyperparameters should be described.

To run any file use:
python3 file_name.py

To train the standard CNN with or without preprocessor use train.py. 
This will also automatically test against noise for the last 10 runs of training.
The model will be saved in the folder Models/ as its time stamp with the bilateral filter amount in front and the noise test results in the folder Results/time/.
m.xlsx will be the mean accuracy over the last 10 runs, full_m.xlsx will show the classwise accuracys, same but with v for the variances over the last 10 training steps.

To train fast use fast.py.
The model will be saved in the folder Models/ as its time stamp with the bilateral filter amount in front.
The code is adapted from: https://github.com/locuslab/fast_adversarial

To test against different noises use noise.py.
The results will be saved in the Results/time/ folder as acc.xlsx

To test against all adversarial attacks except C&W use mult_adv.py.
The restults will be saved in the mult-adv/time/ folder as acc.xlsx.

To test against C&W attacks use CW_adv.py.
The restults will be saved in the CW-adv/time/ folder as acc.xlsx, L2.xlsx distance file is decreaped for models that use the normalized data. 