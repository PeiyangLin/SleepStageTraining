#!/bin/bash
#SBATCH --job-name=Data_FL_Processing
#SBATCH -p q_ai8
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o LOG/job.%j.out
#SBATCH -e LOG/job.%j.err
# Record current host & date.
hostname; date
# Initialize cuda environement.
source ~/.bashrc
conda activate DL

conda info
python DataProcess_FL.py
