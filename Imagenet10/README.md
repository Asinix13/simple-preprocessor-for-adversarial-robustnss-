Use trainimgnet.py to do 5 training runs at once.

The hyperparamter configuration is the same as for the Standard runs of CIFAR-10, check its readme out to get an overview on how to configer them.

The same goes for mult_imgnet.py, it will also test 5 models at once and the hyperparameter config is the same as for the Standard runs.
But it will only test FGSM, APGD(infinity norm), EotPGD and TABPD.

Here the Models are saved together with the config and the test results from mult test is also saved in the same folder under Models/.
