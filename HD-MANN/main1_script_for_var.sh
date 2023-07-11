#!/bin/bash   

#$ -M codell2@nd.edu	# email for notifications
#$ -m a
#$ -l gpu=1  		    # number of gpus
#$ -q gpu	            # queue
#$ -N HD-MANN-cos	    # job name

module load cuda/10.2 cudnn/8.0.4
conda activate env1 
python main1_cos.py
