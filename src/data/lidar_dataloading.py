#import os
#from PIL import Image
#from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as T

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

# The absolute physical maximum height bin from the hypercube
GLOBAL_MAX_HEIGHT = 128.0

def get_chm_pair(data):
    lr = data.get('chm_input', data.get('input'))
    hr = data.get('chm_target', data.get('target'))

    if lr is None or hr is None:
        raise KeyError(f"Missing keys in file. Found: {list(data.keys())}")

    return lr, hr

def load_split_to_ram(split_dir, normalize_diffusion=False):
    """
    Reads all .npz files in a directory into memory, applies global normalization,
    and returns a PyTorch TensorDataset.
    """
    files = sorted([f for f in os.listdir(split_dir) if f.endswith('.npz')])
    
    all_inputs = []
    all_targets = []
    
    print(f"Loading {len(files)} files from {split_dir} into RAM...")
    for f in tqdm(files, desc=f"Caching {os.path.basename(split_dir)}"):
        file_path = os.path.join(split_dir, f)
        data = np.load(file_path)
        
        #lr_chm = data['chm_input']   # Expected shape: (32, 16)
        #hr_chm = data['chm_target']  # Expected shape: (48, 48)
        

        lr_chm, hr_chm = get_chm_pair(data)

        all_inputs.append(lr_chm)
        all_targets.append(hr_chm)
        
    # Convert lists to massive numpy arrays, then to PyTorch tensors
    # Add channel dimension -> [N, 1, H, W]
    inputs_tensor = torch.from_numpy(np.stack(all_inputs)).float().unsqueeze(1)
    targets_tensor = torch.from_numpy(np.stack(all_targets)).float().unsqueeze(1)
    
    # 1. Global Normalization strictly to [0, 1]
    inputs_tensor = torch.clamp(inputs_tensor / GLOBAL_MAX_HEIGHT, min=0.0, max=1.0)
    targets_tensor = torch.clamp(targets_tensor / GLOBAL_MAX_HEIGHT, min=0.0, max=1.0)
    
    # 2. Shift to [-1, 1] for Diffusion models if requested
    if normalize_diffusion:
        inputs_tensor = (inputs_tensor * 2.0) - 1.0
        targets_tensor = (targets_tensor * 2.0) - 1.0
        
    return TensorDataset(inputs_tensor, targets_tensor)

class LazyCHMDataset(Dataset):
    """
    Standard PyTorch Dataset for lazy loading from disk. 
    Use this if your dataset ever grows too large for RAM.
    """
    def __init__(self, split_dir, normalize_diffusion=False):
        self.split_dir = split_dir
        self.files = sorted([f for f in os.listdir(split_dir) if f.endswith('.npz')])
        self.normalize_diffusion = normalize_diffusion

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.split_dir, self.files[idx])
        data = np.load(file_path)
        
        #inp = torch.from_numpy(data['chm_input']).float().unsqueeze(0)
        #tgt = torch.from_numpy(data['chm_target']).float().unsqueeze(0)
        lr_chm, hr_chm = get_chm_pair(data)
        inp = torch.from_numpy(lr_chm).float().unsqueeze(0)
        tgt = torch.from_numpy(hr_chm).float().unsqueeze(0)
        
        # Global Normalization
        inp = torch.clamp(inp / GLOBAL_MAX_HEIGHT, min=0.0, max=1.0)
        tgt = torch.clamp(tgt / GLOBAL_MAX_HEIGHT, min=0.0, max=1.0)
        
        if self.normalize_diffusion:
            inp = (inp * 2.0) - 1.0
            tgt = (tgt * 2.0) - 1.0
            
        return inp, tgt

def build_dataloaders(config, test_only=False, return_test_dataset=False, normalize_diffusion=False, preload_to_ram=False):
    """
    Main entry point to build dataloaders. Matches your previous training scripts.
    """
    root_dir = config.infra.root_dir
    # Support root_dir as a list
    if hasattr(root_dir, "__iter__") and not isinstance(root_dir, (str, bytes)):
        root_dir = list(root_dir)
        existing = [p for p in root_dir if os.path.exists(str(p))]
        root_dir = existing[0] if existing else root_dir[0]
    root_dir = str(root_dir)
    
    batch_size = config.loader.per_gpu_batch_size
    num_workers = 0 if preload_to_ram else config.infra.num_workers_per_gpu
    pin_memory = config.infra.pin_memory
    persistent_workers = False if preload_to_ram else config.infra.persistent_workers
    
    # Helper to choose between RAM Dataset and Lazy Dataset
    def get_dataset(split_name):
        split_path = os.path.join(root_dir, split_name)
        if preload_to_ram:
            return load_split_to_ram(split_path, normalize_diffusion)
        else:
            return LazyCHMDataset(split_path, normalize_diffusion)

    if test_only:
        test_dataset = get_dataset('test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        if return_test_dataset:
            return test_loader, test_dataset
        return test_loader

    else:
        train_dataset = get_dataset('train')
        val_dataset = get_dataset('validation')
        test_dataset = get_dataset('test')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        
        return train_loader, test_loader, val_loader



"""class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform_input=None, transform_target=None):
        
        self.input_dir = os.path.join(root_dir, "inputs")
        self.target_dir = os.path.join(root_dir, "targets")

        self.input_files = sorted(os.listdir(self.input_dir))
        self.target_files = sorted(os.listdir(self.target_dir))

        assert len(self.input_files) == len(self.target_files), \
            "Inputs and targets have different number of files"

        self.transform_input = transform_input
        self.transform_target = transform_target

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])

        inp = Image.open(input_path)#.convert("L")   # grayscale
        tgt = Image.open(target_path)#.convert("L")

        if self.transform_input:
            inp = self.transform_input(inp)
        if self.transform_target:
            tgt = self.transform_target(tgt)

        return inp, tgt

class PairedImageDataset_old(Dataset):
    def __init__(self, root_dir, transform_input=None, transform_target=None, cache=True):
        self.input_dir = os.path.join(root_dir, "inputs")
        self.target_dir = os.path.join(root_dir, "targets")

        self.input_files = sorted(os.listdir(self.input_dir))
        self.target_files = sorted(os.listdir(self.target_dir))

        assert len(self.input_files) == len(self.target_files), \
            "Inputs and targets have different number of files"

        self.transform_input = transform_input
        self.transform_target = transform_target
        self.cache = cache
        self.cached_inputs = []
        self.cached_targets = []

        if self.cache:
            print(f"Caching images from {root_dir} into RAM...")
            for i in range(len(self.input_files)):
                # Load once, keep in memory
                inp = Image.open(os.path.join(self.input_dir, self.input_files[i])).convert("L")
                tgt = Image.open(os.path.join(self.target_dir, self.target_files[i])).convert("L")
                self.cached_inputs.append(inp)
                self.cached_targets.append(tgt)
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        if self.cache:
            inp = self.cached_inputs[idx]
            tgt = self.cached_targets[idx]
        else:
            # Your original loading logic
            inp = Image.open(os.path.join(self.input_dir, self.input_files[idx])).convert("L")
            tgt = Image.open(os.path.join(self.target_dir, self.target_files[idx])).convert("L")

        if self.transform_input:
            inp = self.transform_input(inp)
        if self.transform_target:
            tgt = self.transform_target(tgt)

        return inp, tgt
    
def build_dataloaders(config, test_only=False, return_test_dataset=False, normalize_diffusion=False):
    if normalize_diffusion:
        # For Diffusion models (SR3, Fast Diffusion) that expect [-1, 1]
        sat_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]), # Centers 0.5 to 0.0, range [-1, 1]
        ])
        lidar_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]), # Centers 0.5 to 0.0, range [-1, 1]
        ])
    else:
        # Standard GeSS / Autoregressive behavior: [0, 1]
        sat_transform = T.Compose([
            T.ToTensor(),
        ])
        lidar_transform = T.Compose([
            T.ToTensor(),
        ])

    root_dir = config.infra.root_dir
    # Support root_dir as a list (e.g. multiple machines): use first path that exists, else first
    if hasattr(root_dir, "__iter__") and not isinstance(root_dir, (str, bytes)):
        root_dir = list(root_dir)
        existing = [p for p in root_dir if os.path.exists(str(p))]
        root_dir = existing[0] if existing else root_dir[0]
    root_dir = str(root_dir)
    batch_size=config.loader.per_gpu_batch_size
    num_workers=config.infra.num_workers_per_gpu
    pin_memory=config.infra.pin_memory
    persistent_workers=config.infra.persistent_workers
    
    if test_only: 
        test_dataset = PairedImageDataset(os.path.join(root_dir,'test'),
                                transform_input=sat_transform,
                                transform_target=lidar_transform)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )

        if return_test_dataset:
            return test_loader, test_dataset
        else:
            return test_loader
    else:
        


        train_dataset = PairedImageDataset(os.path.join(root_dir,'train'),
                                    transform_input=sat_transform,
                                    transform_target=lidar_transform)
        
        test_dataset = PairedImageDataset(os.path.join(root_dir,'test'),
                                    transform_input=sat_transform,
                                    transform_target=lidar_transform)
        
        val_dataset = PairedImageDataset(os.path.join(root_dir,'validation'),
                                    transform_input=sat_transform,
                                    transform_target=lidar_transform)
        
        train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                drop_last=True
            )
        
        test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers
            )
        
        val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers
            )
        return train_loader, test_loader ,val_loader"""