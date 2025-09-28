import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class iNF200(Dataset):
    def __init__(self, root='./', train=True,
                 index_path=None, index=None, 
                 base_sess=False, is_clip=False):
 
        self.root = os.path.expanduser(root)
        self.train = train
        self.base_sess = base_sess
        self.data = []
        self.targets = []
        
      
        self._build_label_mapping()
        
        if train:
            if base_sess:
                self._select_by_txt(index_path)
            else:
                self._select_by_txt(index_path)
        else:
            self._select_test_set()

       
        self.transform = self._build_transform(is_clip)

    def _build_label_mapping(self):
        
        self.label_dict = {}
        self.reverse_label_dict = {}
        
       
        data_dir = "train_mini" if self.train else "val"
        base_path = osp.join(self.root, 'iNF200', data_dir)
        
 
        class_folders = sorted(os.listdir(base_path))
        for idx, folder in enumerate(class_folders):
           
            class_name = folder.split('_')[-1]
            self.label_dict[folder] = idx  
            self.reverse_label_dict[idx] = class_name  

    def _select_by_classes(self, class_indices):

        selected_data = []
        
       
        all_classes = sorted(os.listdir(osp.join(self.root, 'iNF200', "train_mini")))
        
        for class_idx in class_indices:
            if class_idx >= len(all_classes):
                raise ValueError(f"Class index {class_idx} out of range")
                
            class_folder = all_classes[class_idx]
            class_path = osp.join(self.root, "train_mini", class_folder)
            
          
            samples = sorted(os.listdir(class_path))[:50]
            for img_name in samples:
                img_path = osp.join(class_path, img_name)
                if osp.exists(img_path):
                    selected_data.append((
                        img_path,
                        self.label_dict[class_folder] 
                    ))

        if not selected_data:
            raise RuntimeError("No data found for selected classes")
            
        self.data, self.targets = zip(*selected_data)

    def _select_by_txt(self, index_path):
        
        with open(index_path, 'r') as f:
            path_lines = [line.strip() for line in f.readlines()]
        
        selected_data = []
        for rel_path in path_lines:
           
            parts = rel_path.split('/')
            if len(parts) != 3:
                continue
                
            class_folder = parts[1]
            img_name = parts[2]
            
            full_path = osp.join(self.root, 'iNF200', rel_path)
            
           
            if class_folder not in self.label_dict:
                continue
                
            if osp.exists(full_path):
                selected_data.append((
                    full_path,
                    self.label_dict[class_folder]
                ))

        if not selected_data:
            print(f"Warning: Empty session {index_path}")
            
        self.data, self.targets = zip(*selected_data) if selected_data else ([], [])

    def _select_test_set(self):
       
        selected_data = []
        val_dir = osp.join(self.root, 'iNF200', "val")
        
        for class_folder in os.listdir(val_dir):
            class_path = osp.join(val_dir, class_folder)
            if not osp.isdir(class_path):
                continue
                
          
            for img_name in os.listdir(class_path):
                img_path = osp.join(class_path, img_name)
                if osp.exists(img_path):
                    selected_data.append((
                        img_path,
                        self.label_dict[class_folder]
                    ))

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

    @property
    def num_classes(self):
        return len(self.label_dict)

if __name__ == '__main__':
   
    base_dataset = iNF200Dataset(
        root="/path/to/iNF200",
        train=True,
        index=list(range(100)), 
        base_sess=True
    )
    
    inc_dataset = iNF200Dataset(
        root="/path/to/iNF200",
        train=True,
        index_path="data/index_list/iNF200/session_2.txt",
        base_sess=False
    )
  
    val_dataset = iNF200Dataset(
        root="/path/to/iNF200",
        train=False
    )
  