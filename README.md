# TLSH + TCAM for accelerating MANN on crossbar arrays

This is the source code for the paper: [Experimentally realized memristive memory augmented neural networks](https://arxiv.org/abs/2204.07429). We implement the training and inference for the full precision model along with the TLSH and TCAM simulation for crossbar arrays. We also reproduce the result in [Robust high-dimensional memory-augmented neural networks](https://www.nature.com/articles/s41467-021-22364-0) in `./HD-MANN` and compare it with ours.

## Code structure

* TLSH_MANN

  `HD-MANN`: Code and model for reproducing the results in High-dimensional MANN

  `configs`: Configs for training the model and inference on the LSH and TLSH + TCAM

  `results`: Model and inference checkpoint

  `memory.py`: Memory module in TLSH_MANN

  `omniglot.py`: Batching training and set sample`

  `cnn.py`: Main training code

  `data_utils.py`: Downloading and preprocessing Omniglot dataset

  `dpe_tcam.py`: Utils for simulating crossbar-based TCAM

  `lib_lsh.py`: Functions for LSH simulation

  `lib_simlsh`: Functions for simulating TLSH

  `simArrayPy.py`: Simulating the IR drop (wire resistance) in crossbar arrays

  `LSHsim.py`: Simulating the LSH and TLSH+TCAM based on the trained model

## Installation

Run following command in your directory:

```shell
git clone https://github.com/Jaylenne/TLSH_MANN.git
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

We provide the configs for training the model in `./configs/trainconfig.config`.You can modify the configs in the `.config`file. The description of the arguments can be found in `./cnn.py`file. To start training the model, run:

```shell
python cnn.py -c ./configs/trainconfig.config
```

### Evaluate the model on crossbar arrays using TLSH + TCAM 

The model is trained first and then can be evaluated on the crossbar arrays using TLSH + TCAM scheme. The configs for evaluating the crossbar behaviors can be found in `./configs/lshconfig.config`. The discription of the arguments can be found in `./LSHsim.py`. To evaluate the results, run:

```shell
python LSHsim.py -c ./configs/lshconfig.config
```

## Experimental results

### Full precision accuracy on different key dimensions

To provide better reproductivity of our results, we provide our trained model in the directory: `./results`. We provide 2 models with different key dimensions: 32 and 512. Specifically, to match the parameters count with the HD method, we use different  `ch_last` in the CNN model. For the model with 32-dimensional output, we use 256 `ch_last`. For the model with 512-dimensional output, we use 128 `ch_last` . The model is tested on 3 tasks: 5-way 1-shot, 20-way 5-shot, 100-way 5-shot and is averaged on 1000 episodes run. We also provide the results of HD-MANN in `./HD-MANN/results` to give a direct comparison.

|      Task      | [32dim](./results/model/32dim/model_best.pth.tar) | [512 dim](./results/model/512dim/model_best.pth.tar) | HD-MANN 32dim                                                | HD-MANN 512dim                                               |
| :------------: | :------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|  5-way 1-shot  | 95.20%                                            | 97.27%                                               | [97.08%](./HD-MANN/results/32dim/5way1shot32dim_best.pth.tar) | [97.74%](./HD-MANN/results/512dim/5way1shot512dim_best.pth.tar) |
| 20--way 5-shot | 97.45%                                            | 98.45%                                               | [97.80%](./HD-MANN/results/32dim/20way5shot32dim_best.pth.tar) | [98.09%](./HD-MANN/results/512dim/20way5shot512dim_best.pth.tar) |
| 100-way 5-shot | 92.90%                                            | 95.01%                                               | [92.59%](./HD-MANN/results/32dim/100way5shot32dim_best.pth.tar) | [94.62%](./HD-MANN/results/512dim/100way5shot512dim_best.pth.tar) |

### TLSH + TCAM accuracy simulated for crossbar arrays

Here we report the ideal LSH using software and TLSH +TCAM simulation with 512 `key_dim`. In the TLSH+TCAM simulation, we consider the **conductance relaxation**, **conductance fluctuation**, and **wire resistance** for $64\times64$ crossbar arrays. The code to simulate the wire resistance in crossbar array is in `./simArrayPy.py` The hashing is based on 32 `key_dim` real-valued vectors. We also provide our inference checkpoint in `./results/LSH_inference`.

| Task           | LSH    | TLSH+TCAM |
| -------------- | ------ | --------- |
| 5-way 1-shot   | 97.82% | 97.64%    |
| 20-way 5-shot  | 97.71% | 97.52%    |
| 100-way 5-shot | 92.39% | 91.56%    |

## Acknowledgement

Part of the code is borrowed from [LSH_Memory](https://github.com/RUSH-LAB/LSH_Memory) for the ICLR paper: [Learning to remember rare events](https://arxiv.org/abs/1703.03129). We thank the open-source implementations. We also thank the collaborators' (Rui Lin) [BAT-MANN](https://github.com/RuiLin0212/BATMANN) and open source code [HD-MANN](https://github.com/DailinH/HD-MANN) for the reproduction of [Robust high-dimensional memory-augmented neural networks](https://www.nature.com/articles/s41467-021-22364-0).

