# Deep learning LLP tagger
This repo highlights the work I have done as part of my undergraduate thesis at UBC. The experiments outlined in my paper will not be reproducible since the datasets are not meant to be shared outside of the ATLAS collaboration. Note there are several scripts that were omitted from this public repo to emphasize the work I contributed to the overall ATLAS analysis and is summarized in this README.

## Intro
Many extensions to the Standard Model (SM) suggest the existence of long lived particles (LLP) that only interact with the SM through a weakly-coupled mediator. The extended lifetime of these particle and the lack of interaction with the SM causes displaced hadronic jets in a detector. Ongoing and published research by the ATLAS collaboration, have presented machine learning models to classify displaced jets in the ATLAS calorimeters. In my thesis, I proposed a modified architecture with the aim to improve the model performance.

## Initial Study
Prior to exploring improved model architectures, an experiment was performed to verify the decision to sort certain input data by transverse momentum (pT) due to the nature of LSTM layers expecting ordered inputs. Three models were trained with the following pT orderered datasets: descending, ascending, and random. 

I modified the preprocessing step [here](https://github.com/rdesc/deep-learning-llp-tagger/blob/master/pre_process.py#L126) to include logic that handles/executes the different pT ordering options. I also made the script [pt_ordering_plot.py](https://github.com/rdesc/deep-learning-llp-tagger/blob/master/pt_ordering_plot.py) to visualize the different pT distributions.

Results from a KFold cross validation showed both the models trained with ascending and descending datasets performing better than the model trained with random datasets.

## Main Contribution
Below is the proposed model architecture which builds off a recurrent deep network.
![](https://github.com/rdesc/deep-learning-llp-tagger/blob/master/thesis_files/new_arch.png)

A hyperparameter grid search was performed to optimize the modified model architecture. Below is a comparison between the proposed model and the previous model. The proposed network attained a relative improvement of 10% in ROC AUC in comparison to the original model. My full thesis report which describes the project in detail can be found here.

| Model  | ROC AUC | Accuracy |
| ------------- | ------------- | ------------- |
| Proposed model  |  0.96  |  0.97  |
| Previous model | 0.87 | 0.94 |
