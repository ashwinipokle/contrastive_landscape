This repository contains the code for all the experiments in 'Contrasting the landscape of contrastive and non-contrastive learning'. This code has been developed colaboratively by Jinjin Tian, Yuchen Li and Ashwini Pokle. 

## Getting Started

### Requirements

Python >= 3.6 and PyTorch >= 1.7. 

To install requirements:

``
$ conda create --name <env> --file requirements.txt
``

### Training

To train models, run any of the following files with appropriate commandline arguments:
- `main.py` to train models with non-contrastive loss as defined in our paper. 
- `main_simclr.py` to train models with variants of contrastive loss, including SimCLR loss and architecture. 
- `main_simsiam.py` to train models with SimSiam loss objective and architecture.

For more concrete examples, check script files provided in `scripts` directory, where we have provided several files used to run experiments included in our paper. 

Several hyperparameters have been included in the files in `config` directory. 

Currenlty, by default, all the logging is done in [wandb](https://wandb.ai/site). Include `--log_metrics` in the command while training the model.

### Bibtex

If you find this work useful for your research, please consider citing out work:

```
@inproceedings{,
  author    = {Ashwini Pokle and Jinjin Tian and Yuchen Li and Andrej Risteski},
  title     = {Contrasting the landscape of contrastive and non=contrastive learning},
  booktitle = {International Conference on Artificial Intelligence and Statistics},
  year      = {2022},
}
```
