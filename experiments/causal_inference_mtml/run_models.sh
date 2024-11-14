#!/bin/bash
#SBATCH -p scavenger-gpu --account=carlsonlab --gres=gpu:1 --mem=64G
#SBATCH --job-name=cmde_mtml
#SBATCH --output=cmde_mtml_%a.out
#SBATCH --error=cmde_mtml_%a.err
#SBATCH -a 1-2
#SBATCH -c 2
#SBATCH --nice

srun --cpu_bind=cores singularity exec --nv --bind /work/zj63 /datacommons/carlsonlab/Containers/multimodal_gp.simg python run_models.py