# TLSH + TCAM for accelerating MANN on crossbar arrays

This is the source code for the paper: [Experimentally realized memristive memory augmented neural networks](https://arxiv.org/abs/2204.07429). We implement the training and inference for the full precision model along with the TLSH and TCAM simulation for crossbar arrays. 

## Installation

Run following command in your directory:

```shell
git clone
pip install -r requirements.txt
```

## Running the code

### Preparation

We use the omniglot dataset in this repo. To download and split the dataset, in your directory, run the code:

```shell
python data_utils.py
```

After downloading, the `train_omni.pkl` and `test_omni.pkl` should be in your directory.

### Train the model

We provide the configs for training the model in `./configs/trainconfig.config`.You can modify the configs in the `.config`file. The discription of the arguments can be found in `./cnn.py`file. To start training the model, run:

```shell
python cnn.py -c ./configs/trainconfig.config
```

### Evaluate the model on crossbar arrays using TLSH + TCAM 

The model is trained first and then can be evaluated on the crossbar arrays using TLSH + TCAM scheme. The configs for evaluating the crossbar behaviors can be found in `./configs/lshconfig.config`. The discription of the arguments can be found in `./LSHsim.py`. To evaluate the results, run:

```shell
python LSHsim.py -c ./configs/lshconfig.config
```

## Experimental results

### Full precision accuracy on different key dimension

To provide better reproductivity of our results, we provide the our trained model in the directory: `./results`. We provide 3 models with different key dimension: 32, 64 and 256. The model is tested on 3 tasks: 5-way 1-shot, 20-way 5-shot, 100-way 5-shot.

|      Task      | [32dim](./results/model/32dim/model_best.pth.tar) | [64 dim](./results/model/64dim/model_best.pth.tar) | [256 dim](./results/model/256dim/model_best.pth.tar) |
| :------------: | :------------------------------------------------ | -------------------------------------------------- | ---------------------------------------------------- |
|  5-way 1-shot  | 93.27%                                            | 96.63%                                             | 97.02%                                               |
| 20--way 5-shot | 97.05%                                            | 97.61%                                             | 97.82%                                               |
| 100-way 5-shot | 92.55%                                            | 93.02%                                             | 92.64%                                               |

### TLSH + TCAM accuracy simulated for crossbar arrays

The figure shows the classification accuracy versus TLSH key dimension on the 100-way 5-shot problem. We compare the simulation (with experimentally validated data) and ideal LSH in software and the full-precision accuracy using 64 key dimension.

![](./results/figure/LSHomni_64keydim_cos_lsh_xbar.svg)

## Acknowledgement

Part of the code is borrowed from [LSH_Memory](https://github.com/RUSH-LAB/LSH_Memory) for the ICLR paper: [Learning to remember rare events](https://arxiv.org/abs/1703.03129). We thank the open-source implementations. We also thank the collaborators' (Rui Lin) [BAT-MANN](https://github.com/RuiLin0212/BATMANN) for reproduction of [HD-MANN](https://www.nature.com/articles/s41467-021-22364-0).