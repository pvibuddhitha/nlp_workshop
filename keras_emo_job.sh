#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=16  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-00:10      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=pkiri056@uottawa.ca
#SBATCH --mail-type=ALL

module load cuda cudnn python/3.6
source ~/tfgpu/bin/activate
# tensorboard --logdir=/tensorlogs/emo --host 0.0.0.0 &
python ./keras_emo_main.py