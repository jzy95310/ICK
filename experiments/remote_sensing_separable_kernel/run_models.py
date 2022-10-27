import os
import sys
sys.path.insert(0, '.')
from subprocess import call

JOB_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
offset = 0

job_dict = {
    1+offset: 'env_data_experiment_CNNRF_seasonal_sorted_by_time.py', 
    2+offset: 'env_data_experiment_ViTRF_seasonal_sorted_by_time.py', 
    3+offset: 'env_data_experiment_DeepViTRF_seasonal_sorted_by_time.py', 
    4+offset: 'env_data_experiment_MAE_ViTRF_seasonal_sorted_by_time.py', 
    5+offset: 'env_data_experiment_CNN_ICKy_sorted_by_time.py', 
    6+offset: 'env_data_experiment_ViT_ICKy_sorted_by_time.py', 
    7+offset: 'env_data_experiment_DeepViT_ICKy_sorted_by_time.py', 
    8+offset: 'env_data_experiment_CNNRF_sorted_by_time.py', 
    9+offset: 'env_data_experiment_ViTRF_sorted_by_time.py'
}

if JOB_ID not in job_dict:
    print(f"{JOB_ID} not here!")
    quit()

call(["python", job_dict[JOB_ID]])