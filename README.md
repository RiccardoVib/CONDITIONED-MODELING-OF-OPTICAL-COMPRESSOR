This code repository for the article _Deep Learning Conditioned Modeling of Optical Compression_ and _Fully Conditioned and Low Latency Black-boc Model of Analog Compression_, Proceedings of the DAFx Conferences 2022-23.

# DEEP LEARNING CONDITIONED MODELING OF OPTICAL COMPRESSION
Datsets are available at the following link:
[CL 1B Dataset (2 parameters)](https://doi.org/10.5281/zenodo.6497085)
 
 
# FULLY CONDITIONED AND LOW LATENCY BLACK-BOX MODEL OF ANALOG COMPRESSION COMPRESSION
 
Visit our [companion page with audio examples](https://riccardovib.github.io/OpticalCompressor_pages/)

This repository contains all the necessary utilities to use our architectures. Find the code located inside the "./Code" folder, and the weights of pre-trained models inside the "./Weights" folder

Visit our [companion page with audio examples](https://riccardovib.github.io/Hybrid-Neural-Audio-Effects_pages/)

### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)
3. [VST Download](#vst-download)

<br/>

# Datasets

Datsets are available at the following links:

[CL 1B Dataset (4 parameters)](https://doi.org/10.34740/kaggle/dsv/5330581)

[PSP MicroComp Dataset (4 parameters)](https://doi.org/10.34740/kaggle/dsv/6118717)

[U-he Presswerk Dataset (4 parameters)](https://doi.org/10.34740/kaggle/dsv/6118202)

[Softube FET Dataset (4 parameters)](https://doi.org/10.34740/kaggle/dsv/6119102)

[LA-2A Dataset (2 parameters)](https://zenodo.org/record/3824876)

# How To Train and Run Inference 

First, install Python dependencies:
```
cd ./requiremnts
pip install -r requirements.txt
```

To train models, use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use. [ [str] ] (default=[" "] )
* --epochs - Number of training epochs. [int] (defaut=60)
* --batch_size - The size of each batch [int] (default=8 )
* --units = The hidden layer size (amount of units) of the network. [ [int] ] (default=8)
* --w_length - The input temporal size [int] (default=16)
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)

Example training case: 
```
cd ./Code/

python starter.py --datasets CL1B --epochs 500 
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./Code/
python starter.py --datasets CL1B --save_folder CL1BModel --only_inference True
```

The repo include two analog (CL1B, LA2A) and three digital (FET, PSP, Presswerk) pre-trained models.


# VST Download

Coming soon...