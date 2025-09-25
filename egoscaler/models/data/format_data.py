from glob import glob
import json
import random
from tqdm import tqdm
import os

def coco_formatter(filepaths, dataset_dir):
    coco_format = {}
    coco_format['info'] = "This dataset contains maximum trajectory data. \
                            Thus, this dataset has noisy data, \
                            requiring some filtering methods."
    coco_format['images'] = []
    coco_format['licenses'] = []
    coco_format['type'] = "instances"
    coco_format['annotations'] = []
    for i, filepath in enumerate(tqdm(filepaths)):
        try:
            with open(f'{dataset_dir}/infos/{filepath}.json', 'r') as f:
                info = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            continue
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {filepath}")
            continue
        
        if info['action_description'] is None: #or info['start_sec'] is None or info['end_sec'] is None:
            continue
        
        coco_format['images'].append(
            {
                "dataset_name": info['dataset_name'],
                "video_uid": info['video_uid'],
                "file_name": info['file_name'],
                "id": i
            }
        )    
        coco_format['annotations'].append(
            {
                "image_id": i,
                "id": i,
                "action_description": info["action_description"]
            }
        )    
    return coco_format    
    

dataset_dir = "/path/to/dataset/EgoScaler"
save_dir = '/home/user/EgoScaler/egoscaler/models/data'

directories = ['egoexo4d']

train_instances = []
for dir_name in directories:
   path_pattern = os.path.join(dataset_dir, 'trajs', dir_name, '*', '*')
   train_instances.extend(glob(path_pattern))

random.seed(0)
random.shuffle(train_instances)

# e.g., validation set 
# split_ratio = 0.8
# split_index = int(len(train_instances) * split_ratio)
# train_instances_split = train_instances[:split_index]
# val_instances_split = train_instances[split_index:]

train = []
for instance in train_instances:
   instance = instance.replace(f'{dataset_dir}/trajs/', '')
   instance = instance.replace('.pkl', '')    
   train.append(instance)
   
# val = []
# for instance in val_instances_split:
#    instance = instance.replace(f'{dataset_dir}/trajs/', '')
#    instance = instance.replace('.pkl', '')    
#    val.append(instance)
        
print('train size', len(train))
# print('val size', len(val)) 
train_data = coco_formatter(train, dataset_dir)
# val_data = coco_formatter(val, dataset_dir)


with open(f'{save_dir}/train.json', 'w') as f:
    json.dump(train_data, f)

# with open(f'{save_dir}/val.json', 'w') as f:
#     json.dump(val_data, f)