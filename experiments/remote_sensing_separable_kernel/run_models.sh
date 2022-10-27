#!/bin/bash
#SBATCH -p carlsonlab-gpu --account=carlsonlab --gres=gpu:1 --mem=64G
#SBATCH --job-name=icky_exp
#SBATCH --output=icky_exp_%a.out
#SBATCH --error=icky_exp_%a.err
#SBATCH -a 1-9
#SBATCH -c 2
#SBATCH --nice

srun singularity exec --nv --bind /work/zj63 /datacommons/carlsonlab/Containers/multimodal_gp.simg python run_models.py