# Scaling OOD Detection

This repository contains the Species dataset, the StreetHazards dataset, and some code for the paper [Scaling Out-of-Distribution Detection for Real-World Settings](https://arxiv.org/abs/1911.11132).

<img align="center" src="species.png" width="500">

__[Download the Species OOD detection dataset here](https://drive.google.com/drive/folders/1j6l7jfGbKL5P5acwKVyktn4y8bWSTeAJ?usp=sharing).__

<img align="center" src="streethazards.gif" width="500">

__[Download the StreetHazards OOD segmentation dataset here](https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar).__

The optional StreetHazards training set is available [here](https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar). Also, the BDD-Anomaly dataset is sourced from the [BDD100K dataset](https://bdd-data.berkeley.edu). Code for the multi-label out-of-distribution detection experiments is available in [this repository](https://github.com/xksteven/multilabel-ood).


## How to use this repository


    git clone --recursive https://github.com/hendrycks/anomaly-seg

    cd anomaly-seg
    mv defaults.py semantic-segmentation-pytorch/config
    mv anom_utils.py semantic-segmentation-pytorch/
    mv dataset.py semantic-segmentation-pytorch/
    mv eval_ood.py semantic-segmentation-pytorch/
    mv create_dataset.py semantic-segmentation-pytorch/
    cd semantic-segmentation-pytorch

    # Place the above download in semantic-segmentation-pytorch/data/
    cd data/
    tar -xvf streethazards_train.tar
    cd ..
    python3 create_dataset.py
    
    # Train pspnet or another model on our dataset
    python3 train.py

    # To evaluate the model on out of distribution test set
    python3 eval_ood.py DATASET.list_val ./data/test.odgt

Note: to run on single gpu please refer to this [issue#3](https://github.com/hendrycks/anomaly-seg/issues/3#issuecomment-574846086).

To evaluate the model performance using a CRF with our code please install

    pip install pydensecrf

The source package is from https://github.com/lucasb-eyer/pydensecrf 


## Evaluation with BDD100K

We cannot reshare the images from BDD100K so please visit [BDD website](https://bdd-data.berkeley.edu/portal.html) to download them.  The images should be from the 10K set of images that they released.

We have shared the labels in the folder called `seg` and part of the process by which we created these labels in `create_bdd_dataset.py`.  To be able to fully utilize these labels one just needs to pattern match the label ids to the image id (they're the same) from our labels to the BDD images.  

Pretrained models weights are availble at this Google drive [link](https://drive.google.com/file/d/1HIQAhX8WIokZpymslUPDmpbUWdYEKEQ3/view?usp=share_link).

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2019anomalyseg,
      title={Scaling Out-of-Distribution Detection for Real-World Settings},
      author={Hendrycks, Dan and Basart, Steven and Mazeika, Mantas and Zou, Andy and Kwon, Joe and Mostajabi, Mohammadreza and Steinhardt, Jacob and Song, Dawn},
      journal={ICML},
      year={2022}
    }
