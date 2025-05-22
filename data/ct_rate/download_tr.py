import os
import time
import shutil
import pandas as pd
from tqdm import tqdm

from huggingface_hub import hf_hub_download


# huggingface
hf_token = '' # your token
repo_id = 'ibrahimhamamci/CT-RATE'
directory_name = 'dataset/train/'
# save directory
local_dir = '/download/ct_rate/'
os.makedirs(local_dir, exist_ok=True)
# resume
start_at = 0
batch_size = 100

data = pd.read_csv(f'./metafiles/train_labels.csv')
for i in tqdm(range(start_at, len(data), batch_size)):
    data_batched = data[i:i+batch_size]
    
    for name in data_batched['VolumeName']:
        folder1 = name.split('_')[0]
        folder2 = name.split('_')[1]
        folder = folder1 + '_' + folder2
        folder3 = name.split('_')[2]
        subfolder = folder + '_' + folder3
        subfolder = directory_name + folder + '/' + subfolder

        for j in range(1000):
            try:
                hf_hub_download(repo_id=repo_id,
                    repo_type='dataset',
                    token=hf_token,
                    subfolder=subfolder,
                    filename=name,
                    local_dir=local_dir,
                    etag_timeout=60,
                )
                break
            except Exception as e:
                print(f"Attempt {j+1} failed: {e}\nCurrent batch: {i//100}.")
                time.sleep(5)

    shutil.rmtree(f'/download/ct_rate/.cache/huggingface/download/dataset/train/')

    