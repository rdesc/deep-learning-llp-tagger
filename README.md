# LLP NN
## A deep learning based LLP tagger

This tool is made to be used from CalRatio ntuples, merging signal, qcd and BIB together, doing basic pre-selection, pre-processing, training and evaluation.

These instructions are to use primarily on **atlas-ml1** which two GPUs. I would advise to do transforming to pre-processing on flashy, instructions are in the master branch of this repo. Use this branch for GPU training only!

## Set up

You should have git, anaconda set up in your current shell.

```bash
git clone --single-branch --branch gpu_training ssh://git@gitlab.cern.ch:7999/fcormier/llp_nn.git
cd llp_nn/python/
conda env create -f conda_train_llp.yml
source startML1.sh
```

## LLPNN\_Runner

This code, **llpNN_runner.py**, is the steering code for the various functions of llp\_nn. Different arguments to control functionality have to be input into **args.txt**.

### Arguments

These are arguments input to **args.txt**.


#### Training

The following two commands need to be executed before each training and should be added to your **~/.bashrc** file
```bash
source ~/llp_nn/python/startML1.sh
conda activate train_llp
```

Now in **args.txt**:

```bash
--doTraining
```

On/off switch so that you run training. It runs **deepJet_train_keras.py** (**gridSearch_train_keras.py** is the old script and is kept as backup purposes), which takes as optional arguments:

* file_name: Name of the .pkl file containing all the data
* model_to_do: Name of the model
* useGPU2: True to use GPU2
* constit_input: ModelInput object for constituents
* track_input: ModelInput object for tracks
* MSeg_input: ModelInput object for muon segments
* jet_input: ModelInput object for jets
* plt_model: True to save model architecture to disk
* frac: Fraction of events to use in file_name
* batch_size: Number of training examples in one forward/backward pass
* reg_value: Value of regularizer term for LSTM
* dropout_value: Fraction of the input units to drop
* epochs: Number of epochs to train the model
* learning_rate: Learning rate
* hidden_fraction: Fraction by which to multiple the dense layers
* kfold: KFold object to do KFold cross validation
 
You can edit these inputs inside **llpNN_runner.py**.

```bash
--file_name=[path_and_name_of_file_with_data_to_train_over]
```

Tell training code where file with data to run over is. This should be the output of pre-processing from master branch.

Since there are two GPUs, if the first one is being used, simply include this argument in **args.txt** to use the second one:

```bash
--useGPU2
```

WARNING: Watch your memory useage! If two trainings are happening at the same time, memory may exceed the 60G available. Use

```bash
free -h
```

on the command line to monitor memory useage. If number under 'Available' column is low, consider reducing the size of your training data.


```bash
--makeFinalPlots
```

On/off flag, makes some outputs plots from the model given.

```bash
--finalPlots_model=[name_of_model]
```

Give the 'name' of the model. Typically training makes a name given variable *model_to_do* in **llpNN_runner.py** and appends some of the variables and time of creation to make a unique filename. Look for name of model under directory *keras_outputs/*.


```bash
--doKFold
```

Performs a KFold cross validation when this flag is added in the **args.txt** file. ROC AUC and accuracy scores are aggregated at the end of the KFold. The method *process_kfold_run* inside **utils.py** is the code that processes the KFold results.

```bash
--doGridSearch
```

Performs a grid search of the specified hyperparameters when this flag is added in the **args.txt** file. Results are processed by the method *process_grid_search_run* inside **utils.py**.
