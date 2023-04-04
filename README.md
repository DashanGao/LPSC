# Improving Privacy-Utility Trade-off in Vertical Federated Learning


--------

### Abstract

Vertical federated learning (VFL) enables an active party possessing labeled data to enhance model performance (utility) by utilizing auxiliary features from passive parties. 
Recently, there has been a growing concern for protecting label privacy against semi-honest passive parties who may surreptitiously deduce private labels from the information shared by active parties. 
Perturbation methods serve as a protection mechanism by employing noisy labels or gradients to update the passive parties' model. 
However, previous perturbation methods sacrifice much utility to protect privacy, leading to an unfavorable trade-off.
This paper proposes Vertical Federated Gradient Boosting (VFGBoost), a model-agnostic approach that utilizes gradient boosting to 1) improve utility by reducing the bias of the active party's local model, and 2) protect label privacy against passive parties by learning residuals instead of labels.
Furthermore, we incorporate adversarial training to protect label privacy. 
We jointly optimize a utility objective to learn residuals and privacy objectives to preserve label privacy.
Experimental results on four real-world datasets substantiate the efficacy of VFGBoost.

--------

### Project structure

The VFGBoost algorithm is implemented in `vertical_fl/fit_VFGBoost.py`

The datasets are saved in `raw_data/`

The running scripts for each dataset are: `run_VFGBoots_{dataset_name}.py`

The checkpoints are saved in `checkpoints/`

The log files are saved in `logging/`

The tensorboard results are saved in `tensorboard/`

