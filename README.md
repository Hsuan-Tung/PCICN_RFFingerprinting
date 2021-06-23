# PCICN_RFFingerprinting


# A Photonic-Circuits-Inspired Compact Network: Toward Real-Time Wireless Signal Classification at the Edge

This repository is the official implementation of [A Photonic-Circuits-Inspired Compact Network: Toward Real-Time Wireless Signal Classification at the Edge]. 

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

<!--- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...--->

## Access to Dataset

The dataset will be available later once it is approved to be publicly released:

<!--- - [Residual dataset of 30 ZigBee devices](https://drive.google.com/drive/folders/1OjTdA0EHeh_DfGZZ8x0Aj8ShiRKw2BH1?usp=sharing) -->
- The residaul data of ZigBee transmissions is pre-processed by Princeton University using the method described in Appendix A. The raw data is provided by Naval Research Laboratory. 
- If you use this dataset for your project, please properly reference [1], [2]\
<a id="1">[1]</a> 
Merchant, Kevin et al. “Deep Learning for RF Device Fingerprinting in Cognitive Communication Networks.” IEEE Journal of Selected Topics in Signal Processing 12 (2018): 160-167.\
<a id="2">[2]</a> 
Peng, Hsuan-Tung et al. "A Photonic-Circuits-Inspired Compact Network: Toward Real-Time Wireless Signal Classification at the Edge", arXiv (2021)

## Training

To train the PRNN-CNN model in the paper, run this command:

```train
python train.py --model PRNN_CNN --data_path your_data_path --save True --output_dir your_output_dir
```

To train the NRL-CNN model in the paper, run this command:

```train
python train.py --model NRL_CNN --data_path your_data_path -save True --output_dir your_output_dir
```

<!-- >📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

## Evaluation

To evaluate the model after training on test dataset, run:

```eval
python eval.py --model_path your_model_path --model_filename your_model.pth --model PRNN_CNN
```
Note that if your model is NRL CNN, please change --model PRNN_CNN to --model NRL_CNN.

<!-- >📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).-->

## Pre-trained Models

To evaluate pretrained PRNN-CNN model on test dataset, run:

```eval
python eval.py --model_path ./pretrained --model_filename output_model_PRNN_CNN_pretrained.pth --model PRNN_CNN
```

To evaluate pretrained NRL-CNN model on test dataset, run:

```eval
python eval.py --model_path ./pretrained --model_filename output_model_NRL_CNN_pretrained.pth --model NRL_CNN
```

## Results

Our model achieves the following performance on :

### [RF Fingerprinting on 30 ZigBee devices]

| Model name         | Top 1 Accuracy  | # Parameters | Estimated Latency (on PYNQ-Z1) |
| ------------------ |---------------- | -------------| -------------------------------|
| NRL CNN            |     95.17%      |   322,602    |            26.19 ms            |
| PRNN-CNN           |     96.32%      |   6,302      |            0.219 ms            |

<!-- >📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. -->

