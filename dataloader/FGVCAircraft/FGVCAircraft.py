# FGVCAircraft.py
import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import sys

class FGVCAircraft(Dataset):
    def __init__(self, root='./', train=True,
                 index_path=None, index=None, 
                 base_sess=None, is_clip=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.data = []
        self.targets = []
        self.label_dict = {}
        # self.label_dict_reverse = {}
        self._pre_process()
        

        if train:
            if base_sess:
                # self._select_by_classes(index)

                self._select_by_txt(index_path)
            else:
                self._select_by_txt(index_path)
        else:
            self._select_by_classes(index)

       
        self.transform = self._build_transform(is_clip)

    def _pre_process(self):
      
        variants_file = osp.join(self.root, 'fgvc-aircraft-2013b/data', 'variants.txt')
        with open(variants_file, 'r') as f:
            self.all_variants = [line.strip() for line in f.readlines()]
        
        
        self.label_dict = {v: idx for idx, v in enumerate(self.all_variants)}
        self.num_classes = len(self.all_variants) 

       
        split_file = 'images_variant_train.txt' if self.train else 'images_variant_test.txt'
        with open(osp.join(self.root, 'fgvc-aircraft-2013b/data', split_file), 'r') as f:
            lines = f.readlines()
        
        self.raw_data = []
        for line in lines:
            parts = line.strip().split(' ', 1)
            if len(parts) != 2:
                continue  
            img_id, variant = parts[0], parts[1]
            
           
            if variant not in self.label_dict:
                raise ValueError(f"Unknown variant '{variant}' detected!")
                
            self.raw_data.append((img_id, variant)) 

    def _select_by_classes(self, index):
        
        selected_variants = [self.all_variants[i] for i in index] 

        selected_data = []
        for img_id, variant in self.raw_data:
            if variant in selected_variants:
                path = osp.join(self.root, 'fgvc-aircraft-2013b/data', 'images', f"{img_id}.jpg")
               
                if osp.exists(path):
                    selected_data.append((
                        path,
                        self.label_dict[variant]  
                    ))
        self.data, self.targets = zip(*selected_data) if selected_data else ([], [])

    def _select_by_txt(self, index_path):

        with open(index_path, 'r') as f:
            path_lines = [line.strip() for line in f.readlines()]

        selected_data = []
        missing_count = 0
        valid_count = 0

       
        id_to_variant = {img_id: variant for img_id, variant in self.raw_data}

        for line in path_lines:
           
            if '/' in line:
                img_id = osp.splitext(osp.basename(line))[0]  
            else:  
                img_id = line.split('.')[0]  

            
            variant = id_to_variant.get(img_id)
            # variant = self.label_dict_reverse(img_id)
            if variant is None:
               
                missing_count += 1
                continue

          
            possible_paths = [
                osp.join(self.root, 'fgvc-aircraft-2013b', line),  
                osp.join(self.root, 'fgvc-aircraft-2013b/data/images', f"{img_id}.jpg"),  
                osp.join(self.root, 'data/images', f"{img_id}.jpg")  
            ]

            found = False
            for path in possible_paths:
                if osp.exists(path):
                    selected_data.append((path, self.label_dict[variant]))
                    valid_count += 1
                    found = True
                    break
            
            if not found:
             
                missing_count += 1

       
        self.data, self.targets = zip(*selected_data) if selected_data else ([], [])

    def _build_transform(self, is_clip):
      
        if is_clip:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        if self.train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, target = self.data[idx], self.targets[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), target