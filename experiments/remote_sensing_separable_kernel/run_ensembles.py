import os
import sys
sys.path.insert(0, '.')
from subprocess import call

JOB_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
offset = 100

job_dict = {
    1+offset: 'env_data_experiment_CNNRF_seasonal_ensemble_temporal.py', 
    2+offset: 'env_data_experiment_ViTRF_seasonal_ensemble_temporal.py', 
    3+offset: 'env_data_experiment_DeepViTRF_seasonal_ensemble_temporal.py', 
    4+offset: 'env_data_experiment_MAE_ViTRF_seasonal_ensemble_temporal.py', 
    5+offset: 'env_data_experiment_CNN_ICKy_ensemble_temporal.py', 
    6+offset: 'env_data_experiment_ViT_ICKy_ensemble_temporal.py', 
    7+offset: 'env_data_experiment_DeepViT_ICKy_ensemble_temporal.py'
}

if JOB_ID not in job_dict:
    print(f"{JOB_ID} not here!")
    quit()

call(["python", job_dict[JOB_ID]])