import torch
import numpy as np
import json
from egoscaler.models.utils.dataset_base import DatasetBase
from egoscaler.configs import DatasetConfig as dataset_cfg
from egoscaler.models.utils.traj_utils import preprocess_traj
import os
from tqdm import tqdm
import re
from egoscaler.models.pointllm.constant import (
    RT2_TOKEN_TEMPLATE, TIMESTEP_SEP_TOKEN, SEP_TOKEN,
    TIMESTEP_START_TOKEN, TIMESTEP_END_TOKEN
)
from egoscaler.models.pointllm.pointllm.data.utils import pc_norm

DESC2TRAJ = {
    "desc": "Action description: {desc}",
    "traj": "To execute the description, action trajectory will be {traj}"
}

class CustomDataset(DatasetBase):
    def __init__(self, args, save_dir, split, tokenizer):
        """
        Initialize the CustomDataset class.

        Parameters:
        - args: Arguments containing dataset configurations.
        - save_dir: Directory to save normalization parameters.
        - split: Dataset split (e.g., 'train', 'val').
        - tokenizer: Tokenizer for encoding descriptions and trajectories.
        """
        super().__init__(args=args, split=split)
        self.args = args
        self.save_dir = save_dir
        self.root_dir = args.root_dir
        self.data_dir = args.data_dir
        self.split = split
        
        self.num_steps = args.num_steps
        self.do_norm = args.do_norm
        self.do_standard = args.do_standard

        assert not (self.do_norm and self.do_standard), "Cannot enable both normalization methods."

        self.tokenizer = tokenizer
        self.max_traj_token = args.max_traj_token
        self.max_desc_token = args.max_desc_token
        
        self.prompt = DESC2TRAJ
        self.eos_token = self.tokenizer.eos_token
        self.sep_token_id = self.tokenizer(SEP_TOKEN, return_tensors='pt', add_special_tokens=False).input_ids
        self.time_sep_token_id = self.tokenizer(TIMESTEP_SEP_TOKEN, return_tensors='pt', add_special_tokens=False).input_ids

        if self.do_standard:
            self._initialize_standardization_params()

    def _initialize_standardization_params(self):
        """
        Initialize standardization parameters (mean and std) based on the dataset split.
        """
        if self.split == "train":
            mean, std = self.compute_mean_std()
            self.save_normalization_params(mean=mean, std=std)
            self.mean, self.std = mean, std
        elif self.split == "val":
            if self.args.debug:
                mean, std = self.compute_mean_std()
                self.save_normalization_params(mean=mean, std=std)
                self.mean, self.std = mean, std
            else:
                self.mean, self.std = self.load_normalization_params()
        else:
            self.mean, self.std = self.load_normalization_params()

    def __len__(self):
        """
        Return the number of annotations in the dataset.
        """
        return len(self.annotations)
    
    def compute_mean_std(self):
        """
        Compute mean and standard deviation for trajectory normalization.

        Returns:
        - mean: Mean values for trajectory dimensions.
        - std: Standard deviation values for trajectory dimensions.
        """
        all_trajs = []
        for item in tqdm(range(len(self.annotations)), desc="Computing mean and std..."):
            _, _, _, traj = super().__getitem__(item=item)
            traj = preprocess_traj(traj, num_steps=self.num_steps)
            all_trajs.append(traj)
            
        all_trajs = np.array(all_trajs)
        mean = all_trajs.mean(axis=(0, 1))
        std = all_trajs.std(axis=(0, 1)) + 1e-8 
        
        return mean, std    
    
    def save_normalization_params(self, mean, std):
        """
        Save normalization parameters (mean and std) to a JSON file.
        """
        params = {'mean': mean.tolist(), 'std': std.tolist()}
        with open(f"{self.save_dir}/norm_param.json", 'w') as f:
            json.dump(params, f)

    def load_normalization_params(self):
        """
        Load normalization parameters (mean and std) from a JSON file.

        Returns:
        - mean: Mean values for trajectory dimensions.
        - std: Standard deviation values for trajectory dimensions.
        """
        with open(f"{self.save_dir}/norm_param.json", 'r') as f:
            params = json.load(f)
        mean = np.array(params['mean'])
        std = np.array(params['std'])
        return mean, std

    def denorm(self, traj: torch.Tensor, max_abs: np.ndarray) -> np.ndarray:
        """
        Denormalize trajectory values based on normalization method.

        Parameters:
        - traj (torch.Tensor): Normalized trajectory tensor.
        - max_abs (np.ndarray): Maximum absolute values for standardization.

        Returns:
        - np.ndarray: Denormalized trajectory.
        """
        traj = traj.detach().cpu().numpy()
        
        if self.do_norm:
            traj[:, :, [0, 1, 2]] = (traj[:, :, [0, 1, 2]] + 1) / 2
            traj[:, :, 0] = traj[:, :, 0] * (dataset_cfg.max_x - dataset_cfg.min_x) + dataset_cfg.min_x
            traj[:, :, 1] = traj[:, :, 1] * (dataset_cfg.max_y - dataset_cfg.min_y) + dataset_cfg.min_y
            traj[:, :, 2] = traj[:, :, 2] * (dataset_cfg.max_z - dataset_cfg.min_z) + dataset_cfg.min_z
            traj[:, :, [3, 4, 5]] *= np.pi
            return traj
        elif self.do_standard:
            traj = traj * max_abs[:, None, :]
            return traj * self.std + self.mean
    
    def collate_fn(self, batch):
        """
        Collate function for batching data.

        Parameters:
        - batch: List of data samples.

        Returns:
        - dict: Batched data.
        """
        image_ids, pcrgbs, desc_tokens, desc_masks, traj_tokens, traj_masks, gt_trajs, gt_traj_masks, max_obs_list = zip(*batch)
                    
        desc_tokens = torch.tensor(desc_tokens, dtype=torch.long)
        desc_masks = torch.tensor(desc_masks, dtype=torch.bool)
        traj_tokens = torch.tensor(traj_tokens, dtype=torch.long)        
        traj_masks = torch.tensor(traj_masks, dtype=torch.bool)
        
        image_ids = torch.stack(image_ids)
        pcrgbs = torch.stack(pcrgbs)
        gt_trajs = torch.stack(gt_trajs)
        gt_traj_masks = torch.stack(gt_traj_masks)
        
        sep_token_id = self.sep_token_id.expand(len(batch), -1)
        seq_token_mask = torch.ones_like(sep_token_id)
        
        prompt = torch.cat((desc_tokens, sep_token_id), dim=-1)
        prompt_mask = torch.cat((desc_masks, seq_token_mask), dim=-1)
        tokens = torch.cat((desc_tokens, sep_token_id, traj_tokens), dim=-1)
        masks = torch.cat((desc_masks, seq_token_mask, traj_masks), dim=-1)
        
        pos = torch.where(tokens[0] == self.time_sep_token_id)[1][0]
        prompt = tokens[:, :pos+1]
        prompt_mask = masks[:, :pos+1]

        return {
            'image_ids': image_ids, 
            'pcrgbs': pcrgbs, 
            'prompts': prompt,
            'prompt_masks': prompt_mask,
            'tokens': tokens,
            'attention_masks': masks,
            'trajectories': gt_trajs,
            'trajectory_masks': traj_masks,
            'max_abs': torch.stack([torch.tensor(ma) for ma in max_obs_list])
        }
        
    def detokenize_traj(self, traj_str: str, num_bins: int = 256, max_abs: np.ndarray = None) -> np.ndarray:

        # Remove timestep start and end tokens
        if traj_str.startswith(TIMESTEP_START_TOKEN):
            traj_str = traj_str[len(TIMESTEP_START_TOKEN):]
        if traj_str.endswith(TIMESTEP_END_TOKEN):
            traj_str = traj_str[:-len(TIMESTEP_END_TOKEN)]

        # Split by timestep
        time_steps = traj_str.split(TIMESTEP_SEP_TOKEN)

        # Build regex pattern to extract discretized integer from RT2_TOKEN_TEMPLATE
        pattern_str = re.escape(RT2_TOKEN_TEMPLATE).replace(re.escape("{p}"), r"([-0-9]+)")
        pattern = re.compile(pattern_str)

        traj_normalized = []
        for ts in time_steps:
            # Extract 6 discretized values per timestep
            matches = pattern.findall(ts)
            if len(matches) != 6:
                continue

            coords = []
            for d_str in matches:
                d = int(d_str)
                # Recover bin center in normalized [-1,1] range
                val = -1 + (d + 0.5) * (2 / num_bins)
                coords.append(val)

            traj_normalized.append(coords)

        traj_normalized = np.array(traj_normalized)
        if traj_normalized.shape[0] == 0:
            return None

        # Inverse normalization
        if self.do_norm:
            # Position: [-1,1] → [0,1] → original scale
            pos = traj_normalized[:, :3]
            pos = (pos + 1) / 2
            pos[:, 0] = pos[:, 0] * (dataset_cfg.max_x - dataset_cfg.min_x) + dataset_cfg.min_x
            pos[:, 1] = pos[:, 1] * (dataset_cfg.max_y - dataset_cfg.min_y) + dataset_cfg.min_y
            pos[:, 2] = pos[:, 2] * (dataset_cfg.max_z - dataset_cfg.min_z) + dataset_cfg.min_z

            # Rotation: multiply by π to restore angle range
            rot = traj_normalized[:, 3:] * np.pi

            traj_rec = np.concatenate([pos, rot], axis=-1)

        elif self.do_standard:
            if max_abs is None:
                raise ValueError("max_abs must be provided when using do_standard.")

            # First undo max_abs scaling, then undo standardization
            traj_rec = traj_normalized * max_abs
            traj_rec = traj_rec * self.std + self.mean

        return traj_rec


    def get_padding(self, tokens, max_length, pad_token_id=0):
        len_input_ids = len(tokens)
        if len_input_ids > max_length:
            tokens = tokens[:max_length]
            len_input_ids = max_length 
        tokens += [pad_token_id] * (max_length - len_input_ids)
        attention_mask = [1] * len_input_ids + [0] * (max_length - len_input_ids)
        
        return tokens, attention_mask
        
    def __getitem__(self, item):
        
        image_id, pil_image, desc, traj = super().__getitem__(item=item)
        
        data: dict = self.id2data[image_id.item()]
        dataset_name: str = data['dataset_name']
        video_uid: str = data['video_uid']
        file_name: str = data['file_name']
        
        pcrgb = np.load(f'{self.root_dir}/pcrgbs/{dataset_name}/{video_uid}/{file_name}.npy')
        pcrgb = pc_norm(pcrgb)
        pcrgb = torch.from_numpy(pcrgb).to(torch.float32)
        
        if self.split in ["h2o_test", "hot3d_test"]:
            traj[:, 3:] -= traj[0, 3:]
            
        traj[:, 3:] = (traj[:, 3:] + np.pi) % (2 * np.pi) - np.pi
        
        traj, traj_mask = preprocess_traj(
            traj, num_steps=self.num_steps, 
            return_padding_mask=True
        )
            
        gt_traj = torch.from_numpy(traj).float()
        gt_traj_mask = torch.from_numpy(traj_mask).bool()
        
        max_abs = np.zeros([traj.shape[1]])
        
        if self.do_norm:
            
            traj[:, 0] = (traj[:, 0] - dataset_cfg.min_x) / (dataset_cfg.max_x - dataset_cfg.min_x)
            traj[:, 1] = (traj[:, 1] - dataset_cfg.min_y) / (dataset_cfg.max_y - dataset_cfg.min_y)
            traj[:, 2] = (traj[:, 2] - dataset_cfg.min_z) / (dataset_cfg.max_z - dataset_cfg.min_z)
            
            traj[:, [0,1,2]] = 2 * traj[:, [0,1,2]] - 1
            
            traj[:, [3,4,5]] /= np.pi # [-1, 1]
            
        elif self.do_standard:
            traj = (traj - self.mean) / self.std
            
            max_abs = np.max(np.abs(traj), axis=0) 
            max_abs[max_abs == 0] = 1  
            
            traj = traj / max_abs  # [-1, 1]
            
        assert np.all(traj >= -1) and np.all(traj <= 1)
            
        traj_list = []
        for tra in traj:
            (x, y, z, rx, ry, rz) = self.discretize_action(tra, num_bins=self.args.num_bins)
            
            x, y, z, rx, ry, rz = (
                RT2_TOKEN_TEMPLATE.format(p=x),
                RT2_TOKEN_TEMPLATE.format(p=y),
                RT2_TOKEN_TEMPLATE.format(p=z),
                RT2_TOKEN_TEMPLATE.format(p=rx),
                RT2_TOKEN_TEMPLATE.format(p=ry),
                RT2_TOKEN_TEMPLATE.format(p=rz)
            )
            coord = f"{x} {y} {z} {rx} {ry} {rz}"
            traj_list.append(coord)
    
        traj_str = f"{' '+TIMESTEP_SEP_TOKEN+' '}".join(traj_list)
        traj_str = TIMESTEP_START_TOKEN + f"{' '+traj_str+' '}" + TIMESTEP_END_TOKEN
        
        desc = self.prompt["desc"].format(desc=desc)
        traj_str = self.prompt["traj"].format(traj=traj_str)
        
        desc_tokens = self.tokenizer.encode(desc, add_special_tokens=False) + \
                    [self.tokenizer.eos_token_id]
        traj_tokens = self.tokenizer.encode(traj_str, add_special_tokens=False) + \
                    [self.tokenizer.eos_token_id]
         
        desc_tokens, desc_masks = self.get_padding(desc_tokens, self.max_desc_token, pad_token_id=self.tokenizer.pad_token_id)
        traj_tokens, traj_masks = self.get_padding(traj_tokens, self.max_traj_token, pad_token_id=self.tokenizer.pad_token_id)
        
        return image_id, pcrgb, desc_tokens, desc_masks, traj_tokens, traj_masks, gt_traj, gt_traj_mask, max_abs