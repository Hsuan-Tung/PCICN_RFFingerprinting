# PCICN_RFFingerprinting


# A Photonic-Circuits-Inspired Compact Network: Toward Real-Time Wireless Signal Classification at the Edge

This repository is the official implementation of [A Photonic-Circuits-Inspired Compact Network: Toward Real-Time Wireless Signal Classification at the Edge]. 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Access to Dataset

You can download dataset here:

- [Residual dataset of 30 ZigBee devices](https://drive.google.com/drive/folders/1NJBWN4dlSAn_uLX7CIYUnB2zCTFaZ98k?usp=sharing) 


## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> 
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on test dataset, run:

```eval
python eval.py --model-file mymodel.pth 
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [Pretrained NRL-CNN and PRNN-CNN models](https://drive.google.com/drive/folders/11ysSUTBOapH_4xOI8-GXRy1MH6lcnOxk?usp=sharing) trained on residual training dataset using parameters described in our manuscript. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [RF Fingerprinting on 30 ZigBee devices]%(https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | # Parameters | Estimated Latency (on PYNQ-Z1) |
| ------------------ |---------------- | -------------| -------------------------------|
| NRL CNN            |     95.17%      |   322,602    |            26.19 ms            |
| PRNN-CNN           |     96.32%      |   6,302      |            0.219 ms            |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

