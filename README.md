# Readily-interpretable deep learning translation of GWAS and multi-omics findings to understand pathobiology and drug repurposing in Alzheimer's disease

Manuscript link: https://www.biorxiv.org/content/10.1101/2021.10.20.465087v1.abstract

Default parameters in the main.py file are the ones used for results mentioned in the manuscript.

## Requirments:
* Python 3.7
* pytorch=1.11.0
* scipy=1.6.3
* numpy=1.20.2
* networkx=2.5.1
* sklearn=0.24.2


## Installation

* conda create -n nettag python=3.7

* conda activate nettag

* pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

* pip install networkx

* pip install spicy

* pip install -U scikit-learn

* pip install pandas

## Usage

The pretrained model used for results in the manuscript is in output folder.

* Not using pertained model: python main.py --hidden_size 2048 1024

* Using pertained model: python main.py --hidden_size 2048 1024 --pretrained True

`
