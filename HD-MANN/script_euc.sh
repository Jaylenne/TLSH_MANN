#!/bin/bash

#$ -M ahennes3@nd.edu	 # Email address for job notification
#$ -m a		 # Send mail when job begins, ends and aborts
#$ -t 1 #-20:1
#$ -l gpu=1  		# specify number of gpus
#$ -q gpu	# specify queue
#$ -N HD-MANN-var	# specify name of the job

module load cuda/10.2 cudnn/8.0.4

conda activate mann-env

python main_euc_quantized.py
