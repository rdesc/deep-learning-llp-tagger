# LLP NN
## A deep learning based LLP tagger

This tool is made to be used from CalRatio ntuples, merging signal, qcd and BIB together, doing basic pre-selection, pre-processing, training and evaluation.

These instructions are to use primarily on **atlas-ml1** which two GPUs. I would advise to do transforming to pre-processing on flashy, instructions are in the master branch of this repo. Use this branch for GPU training only!

## Set up

You should have git, anaconda set up in your current shell.

```bash
git clone --single-branch --branch gpu_training ssh://git@gitlab.cern.ch:7999/fcormier/llp_nn.git
cd llp_nn/python/
mkdir plots/
mkdir keras_outputs/
conda env create -f conda_train_llp.yml
source startML1.sh
```



## LLPNN\_Runner

This code, **llpNN_runner.py**, is the steering code for the various functions of llp\_nn. Different arguments to control functionality have to be input into **args.txt**.

### Arguments

These are arguments input to **args.txt**.


#### Training

To set up for training on ml1 GPUs:

```bash
cd python/
conda activate train_llp
source startML1.sh
```

Need to do this every new login. Now in **args.txt**:


```bash
--doTraining
```

On/off switch so that you run training. It runs **gridSearch_train_keras.py**, which takes as optional arguments:

* Fraction of events
* Number of constituents input
* Number of tracks input
* Number of muon segments input
* How many constituent LSTM nodes
* How many track LSTM nodes
* How many muon segment LSTM nodes
* Regularization value
* Dropout value
* Number of epochs
* name of model
* True/False add tracking to training
* True/False add muon segments to training
* True/False do parametrized training
* learning rate
 
You can edit what is input looking at **llpNN_runner.py**.

```bash
--file_name=[path_and_name_of_file_with_data_to_train_over]
```

Tell training code where file with data to run over is. This should be the output of pre-processing from master branch.

```bash
--makeFinalPlots
```

On/off flag, makes some outputs plots from the model given.

```bash
--finalPlots_model=[name_of_model]
```

Give the 'name' of the model. Typically training makes a name given variable *model_to_do* in **llpNN_runner.py** and appends some of the variables and time of creation to make a unique filename. Look for name of model under directory *keras_outputs/*.
