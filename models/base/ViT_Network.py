import torch
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet18_cifar import resnet18_cifar
from utils import identify_importance
import numpy as np
import copy
# 这里把timm给注释掉
# import timm
from models.vision_transformer import VisionTransformer
import open_clip as clip

# 下面是我把timm给拿出来了
from models import model_My
# from models import vision_transformer_My
from timm.models import create_model

# from .helper import test
from utils import *
# from models.prompt import Global_Prompt_Extractor
import torchvision.transforms as transforms
import os

import random
from models.utils import sample

from models.prompt import Prompt

from models.utils import AddNoise
from models.Prompt_FFN import KeyFFN

import matplotlib.pyplot as plt
import sys

from models.utils import AdaptiveFrequencyMask

class ViT_MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        if self.args.dataset in ['cifar100']:
            self.num_features = 768
        if self.args.dataset in ['mini_imagenet']:
            self.num_features = 768
        if self.args.dataset in ['FGVCAircraft']:
            self.num_features = 768
        if self.args.dataset in ['iNF200']:
            self.num_features = 768
        if self.args.dataset == 'cub200' or self.args.dataset == 'air':
            self.num_features = 768
        
        if args.scratch:
            self.encoder = create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=args.num_classes,
                drop_rate=0.,
                drop_path_rate=0.,
                drop_block_rate=None,
            )
            print("1")
        else:
            self.encoder = create_model("vit_base_patch16_224_in21k",pretrained=True,num_classes=args.num_classes,
                                drop_rate=0.,drop_path_rate=0.,drop_block_rate=None)
            # print("2")
        
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fc.is_classifier = True
        
        self.seen_classes = args.base_class  # 用来输出样本编号或者名称

        self.way = args.way # 这里只拿当前阶段的logits
        self.base_class = args.base_class

        # 下面是Revisiting中的prompt参数
        self.VPT_type = args.VPT_type
        self.Prompt_Token_num = args.Prompt_Token_num
        self.VPT = args.VPT  # 是否使用prompt-deep
        
        if self.VPT_type == "deep": # 选择是否使用Revisiting论文中的prompt
            self.Prompt_Tokens = nn.Parameter(torch.zeros(len(self.encoder.blocks), self.Prompt_Token_num, self.encoder.embed_dim))
            torch.nn.init.normal_(self.Prompt_Tokens, mean=0, std=0.1)
            # torch.nn.init.uniform_(self.Prompt_Tokens, -1, 1)
        else:  # "Shallow"
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, self.Prompt_Token_num, self.encoder.embed_dim))
            torch.nn.init.normal_(self.Prompt_Tokens, mean=0, std=1)

        # # 下面是InsVP里提取glabal prompt的参数
        self.InsVP_prompt_patch = args.InsVP_prompt_patch # 16

        self.meta_dropout_2 = torch.nn.Dropout(self.args.Dropout_Prompt)  # 前面mask的
        self.meta_dropout_3 = torch.nn.Dropout(self.args.Dropout_Block)  # 后面Block的

        self.InsVP_prompt_patch_2 = args.InsVP_prompt_patch_2 # 11 卷积核大小
        self.InsVP_prompt_patch_22 = args.InsVP_prompt_patch_22 # 25
        n_2 = self.InsVP_prompt_patch_2
        n_22 = self.InsVP_prompt_patch_22 # 左下分支

        self.scale_noise = 0.1
        self.meta_net_2 = nn.Sequential( 
            nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
            nn.ReLU(),
            # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
            nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
        )
        if self.args.pixel_prompt == "YES":
            # 设置共11个网络
            if self.args.First_Pool_Prompt_Net0 == "YES":
                self.First_Pool_Prompt_Net0 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )
            if self.args.First_Pool_Prompt_Net1 == "YES":
                self.First_Pool_Prompt_Net1 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )
            if self.args.First_Pool_Prompt_Net2 == "YES":
                self.First_Pool_Prompt_Net2 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )
            if self.args.First_Pool_Prompt_Net3 == "YES":
                self.First_Pool_Prompt_Net3 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )
            if self.args.First_Pool_Prompt_Net4 == "YES":
                self.First_Pool_Prompt_Net4 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )
            if self.args.First_Pool_Prompt_Net5 == "YES":
                self.First_Pool_Prompt_Net5 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )
            if self.args.First_Pool_Prompt_Net6 == "YES":
                self.First_Pool_Prompt_Net6 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )
            if self.args.First_Pool_Prompt_Net7 == "YES":
                self.First_Pool_Prompt_Net7 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )
            if self.args.First_Pool_Prompt_Net8 == "YES":
                self.First_Pool_Prompt_Net8 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )
            if self.args.First_Pool_Prompt_Net9 == "YES":
                self.First_Pool_Prompt_Net9 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )   
            if self.args.First_Pool_Prompt_Net10_23 == "YES":
                self.First_Pool_Prompt_Net10 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )     
                self.First_Pool_Prompt_Net11 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net12 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net13 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net14 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net15 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net16 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net17 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net18 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )         
                self.First_Pool_Prompt_Net19 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net20 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )         
                self.First_Pool_Prompt_Net21 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net22 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )    
                self.First_Pool_Prompt_Net23 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )         
                self.First_Pool_Prompt_Net24 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net25 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )         
                self.First_Pool_Prompt_Net26 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )      
                self.First_Pool_Prompt_Net27 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )    
                self.First_Pool_Prompt_Net28 = nn.Sequential( 
                    nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                    nn.ReLU(),
                    # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                    nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                )   
                # self.First_Pool_Prompt_Net29 = nn.Sequential( 
                #     nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                #     nn.ReLU(),
                #     # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                #     nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                # )  
                # self.First_Pool_Prompt_Net30 = nn.Sequential( 
                #     nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                #     nn.ReLU(),
                #     # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                #     nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                # )  
                # self.First_Pool_Prompt_Net31 = nn.Sequential( 
                #     nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                #     nn.ReLU(),
                #     # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                #     nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                # )  
                # self.First_Pool_Prompt_Net32 = nn.Sequential( 
                #     nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                #     nn.ReLU(),
                #     # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                #     nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                # )  
                # self.First_Pool_Prompt_Net33 = nn.Sequential( 
                #     nn.Conv2d(3, args.InsVP_hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
                #     nn.ReLU(),
                #     # AddNoise(scale=self.scale_noise),  # 添加自定义的扰动层
                #     nn.Conv2d(args.InsVP_hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
                # )  

    
        # 在channel维度进行1x1
        self.meta_net_3 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=1, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        if self.args.Block_prompt == "YES":
            # a=1
            self.meta_net_block_0 = nn.Sequential(
                nn.Conv1d(in_channels=768, out_channels=1, kernel_size=1, stride=1),
                nn.ReLU(),
            )
            self.meta_net_block_1 = nn.Sequential(
                nn.Conv1d(in_channels=768, out_channels=1, kernel_size=1, stride=1),
                nn.ReLU(),
            )
            self.meta_net_block_2 = nn.Sequential(
                nn.Conv1d(in_channels=768, out_channels=1, kernel_size=1, stride=1),
                nn.ReLU(),
            )
            self.meta_net_block_3 = nn.Sequential(
                nn.Conv1d(in_channels=768, out_channels=1, kernel_size=1, stride=1),
                nn.ReLU(),
            )

            # # 在channel维度进行1x1
            # self.meta_net_3 = nn.Sequential(
            #     nn.Conv1d(in_channels=768, out_channels=128, kernel_size=1, stride=1),
            #     nn.BatchNorm1d(128),  # 添加批归一化
            #     nn.ReLU(),
            #     nn.Conv1d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
            #     # nn.BatchNorm1d(64),
            #     nn.ReLU(),
            # )

            # # 在channel维度进行1x1
            # self.meta_net_3 = nn.Sequential(
            #     nn.Conv2d(
            #         in_channels=1,  # 通道数固定为 1，因为我们只在原通道维度上卷积
            #         out_channels=1,  # 输出维度，决定最终特征图的数量
            #         kernel_size=(1, 768),  # 核大小覆盖整个通道维度
            #         stride=(1, 1)
            #     ),
            #     nn.ReLU(),
            # )

            
            # # 二维卷积1x1
            # self.meta_net_3 = nn.Sequential( 
            #     nn.Conv2d(3, 1, 1, stride=1, padding=0),
            #     nn.ReLU(),
            # )

        # 使用高斯分布生成特征
        self.class_mean_list = []
        self.class_cov_list = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # l2p参数
        self.prompt_pool = self.args.prompt_pool
        self.use_prompt_mask = self.args.use_prompt_mask # 这个没有传递给prompt，而是把embed传入了
        
        if  self.args.prompt_pool: 
            # print("使用l2p")
            self.prompt_l2p = Prompt(embed_dim=self.encoder.embed_dim, prompt_pool=self.args.prompt_pool, prompt_key_init=self.args.prompt_key_init,
                                embedding_key=self.args.embedding_key, pool_size=self.args.pool_size, top_k=self.args.top_k, batchwise_prompt=self.args.batchwise_prompt,)

        if self.args.Frequency_mask:
           
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            max_radius = torch.sqrt(torch.tensor((224 / 2) ** 2 + (224 / 2) ** 2)).item()  # 对角线
           
            radii_init = torch.linspace(0, max_radius, steps=self.args.num_r)
            self.radii = radii_init  # 半径为固定值，不再作为参数优化
            
            weights_init = torch.normal(mean=0, std=10, size=(self.args.num_r,))
            self.weights = nn.Parameter(weights_init)  # 设置为可优化的参数

            if self.args.test1:
                self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))
                self.beta = nn.Parameter(torch.tensor(0.5, requires_grad=True))

        self.FFN_input_30prompts = self.args.FFN_input_30prompts
        if self.args.FFN_input_30prompts:
            # 定义两个FFN网络
            self.ffn_input = KeyFFN(embed_dim=3, hidden_dim=6)  
            self.ffn_30prompts = KeyFFN(embed_dim=3, hidden_dim=6)
    #todo =======================================================================================
    
    def update_seen_classes(self, new_classes):
        print('new classes for this session:\n', new_classes)
        self.mask = torch.zeros(self.args.num_classes,device='cuda')
        self.mask[:self.seen_classes]=-torch.inf
        self.seen_classes += len(new_classes)
    
    def encode(self, x):
        x = self.encoder.forward_features(x)[:,0]
        return x
    
    def prompt_encode(self, img, prompt_feat=False, B_tuning=False, eval=False):
        x = self.encoder.patch_embed(img) # (batch_size, 196, embed_dim)
        ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)

        if(self.VPT == "NO"):
            x = self.encoder.blocks(x)
            x = self.encoder.norm(x)  # 多加了一个norm
            cls_embed = x[:,0]
            return cls_embed
        elif (self.VPT == "YES"):
            # 注意是小写
            if self.VPT_type == "deep":
                Prompt_Token_num = self.Prompt_Tokens.shape[1]
                # [depth, Prompt_Token_num, emb]

                for i in range(len(self.encoder.blocks)):
                    # print(len(self.encoder.blocks)) # 12
                    # concatenate Prompt_Tokens
                    # 拿出第i个并增加一个维度[1, Prompt_Token_num, emb]
                    Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                    # firstly concatenate
                    # expand之后是[batch_size, Prompt_Token_num, emb]，之后和x进行cat
                    x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                    # 此时x的维度是[batch_size, seq_len + Prompt_Token_num, emb]
                    num_tokens = x.shape[1]
                    # lastly remove, a genius trick
                    x = self.encoder.blocks[i](x)[:, :num_tokens - Prompt_Token_num]
                    # print(x.shape) # [64,197,768]
                    # 通过这一层，再在处理结束时移除它们

                    if self.args.Block_prompt == "YES":
                        # if i == 0:
                        #     prompt = self.PatchPrompts_Block(x, self.meta_net_block_0)
                        # elif i == 1:
                        #     prompt = self.PatchPrompts_Block(x, self.meta_net_block_1)
                        # elif i == 2:
                        #     prompt = self.PatchPrompts_Block(x, self.meta_net_block_2)
                        # elif i == 3:
                        #     prompt = self.PatchPrompts_Block(x, self.meta_net_block_3)
                        # x = x + prompt

                        # if i == 4:
                        #     prompt = self.PatchPrompts_Block(x, self.meta_net_block_0)
                        #     x = x + prompt
                        # if i == 5:
                        #     prompt = self.PatchPrompts_Block(x, self.meta_net_block_1)
                        #     x = x + prompt
                        # if i == 6:
                        #     prompt = self.PatchPrompts_Block(x, self.meta_net_block_2)
                        #     x = x + prompt
                        # if i == 7:
                        #     prompt = self.PatchPrompts_Block(x, self.meta_net_block_3)
                        #     x = x + prompt

                        alpha = 1
                        if i == 8:
                            prompt = self.PatchPrompts_Block(x, self.meta_net_block_0)
                            x = x + prompt * alpha
                        elif i == 9:
                            prompt = self.PatchPrompts_Block(x, self.meta_net_block_1)
                            x = x + prompt * alpha
                        elif i == 10:
                            prompt = self.PatchPrompts_Block(x, self.meta_net_block_2)
                            x = x + prompt * alpha
                        elif i == 11:
                            prompt = self.PatchPrompts_Block(x, self.meta_net_block_3)
                            x = x + prompt * alpha

            else:  # self.VPT_type == "Shallow"
                Prompt_Token_num = self.Prompt_Tokens.shape[1]

                # concatenate Prompt_Tokens
                # expand之后是[batch_size, Prompt_Token_num, emb]
                Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
                x = torch.cat((x, Prompt_Tokens), dim=1)
                num_tokens = x.shape[1]
                # Sequntially procees
                x = self.encoder.blocks(x)[:, :num_tokens - Prompt_Token_num]
                # 通过之后的每一层，再在处理结束时移除它们

            # print('哈哈')
            # print(Prompt_Tokens)
            x = self.encoder.norm(x)
            x=x[:, 0, :]
            return x
    
    def PatchPrompts_Block(self, x, block):
        x = x.permute(0, 2, 1)  # 调整维度
        x = block(x)  # 使用传入的 block 进行处理
        x = x.permute(0, 2, 1)  # 调整回原始维度
        return self.meta_dropout_3(x)

    def PatchPrompts_Conv1d_1x1(self, x):
        x = x.permute(0, 2, 1)
        x = self.meta_net_3(x) # 拿到每个patch的prompt
        x = x.permute(0, 2, 1)
        return self.meta_dropout_3(x)
    
    def PatchPrompts_Conv1d_1x1_two(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, 196, embed_dim)
        x = self.meta_net_3(x) # 拿到每个patch的prompt
        x = x.squeeze(1)   # (batch_size, 196, 1)
        return self.meta_dropout_3(x)

    def PatchPrompts_Conv2d_1x1(self, x):
        # [64, 3, 224, 224]
        B = x.shape[0]
        n = self.InsVP_prompt_patch
        n_patch = int(224 / n)
        x = x.reshape(B, 3, n_patch, n, n_patch, n) # [64, 3, 14, 16, 14, 16]
        x = x.permute(0, 2, 4, 1, 3, 5) # [64, 14, 14, 3, 16, 16]
        x = x.reshape(B, n_patch*n_patch, 3, n, n)
        x = x.reshape(B*n_patch*n_patch, 3, n, n)
        x = self.meta_net_3(x) # 拿到每个patch的prompt
    
        x = x.reshape(B, n_patch, n_patch, 1, n, n)
        x = x.permute(0, 3, 1, 4, 2, 5) # [64, 3, 14, 16, 14, 16]
        x = x.reshape(B, 1, 224, 224) # 再给合并
        return self.meta_dropout_3(x)
    
    # 定义一个函数计算余弦相似度，更关注局部的
    def cosine_similarity(self, a, b):
        # 归一化向量
        a_norm = F.normalize(a, dim=1)  # [batch_size, channels, h, w]
        b_norm = F.normalize(b, dim=1)  # [batch_size, channels, h, w]
        # 计算点积
        return torch.sum(a_norm * b_norm, dim=1, keepdim=True)  # [batch_size, 1, h, w]
    
    def get_prompts(self, x, session=-1):
        res = {}  # 初始化 res 为一个空字典
        if self.args.pixel_prompt == "YES":
            # 定义 10 个 prompts
            prompts_list = [
                self.meta_dropout_2(self.meta_net_2(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net0(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net1(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net2(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net3(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net4(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net5(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net6(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net7(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net8(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net9(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net10(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net11(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net12(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net13(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net14(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net15(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net16(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net17(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net18(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net19(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net20(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net21(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net22(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net23(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net24(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net25(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net26(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net27(x)),
                self.meta_dropout_2(self.First_Pool_Prompt_Net28(x)),
                # self.meta_dropout_2(self.First_Pool_Prompt_Net29(x)),
                # self.meta_dropout_2(self.First_Pool_Prompt_Net30(x)),
                # self.meta_dropout_2(self.First_Pool_Prompt_Net31(x)),
                # self.meta_dropout_2(self.First_Pool_Prompt_Net32(x)),
                # self.meta_dropout_2(self.First_Pool_Prompt_Net33(x)),
            ]
    
            if self.args.prompt_pool:
                # print("使用l2p")
                with torch.no_grad():
                    output = self.encoder.forward_features(x)[:,0]
                    # 我不用额外定义一个网络，因为l2p中的forward函数需要额外经历prompt
                    cls_features = output

                res = self.prompt_l2p(x, cls_features=cls_features)
                weights = res['weights']  # [batch_size, Pool_size]

                # print(round(weights[0].max().item(), 2), round(weights[0].min().item(), 2))
                # print(round(weights[1].max().item(), 2), round(weights[1].min().item(), 2))
                # print(round(weights[2].max().item(), 2), round(weights[2].min().item(), 2))
                # print()

                if session == 10:
                    # 遍历第一个维度 (shape[0])，对于每个样本
                    for i in range(weights.shape[0]):
                        # 获取每个样本的最大值和最小值，并保留两位小数
                        max_val = round(weights[i].max().item(), 2)
                        min_val = round(weights[i].min().item(), 2)
                        # 输出最大值和最小值
                        print(f"Sample {i}: Max = {max_val}, Min = {min_val}")

                prompts_tensor = torch.stack(prompts_list, dim=0)  # [num_prompts, B, channel, h, w]
                prompts_tensor = prompts_tensor.permute(1, 0, 2, 3, 4)  # [B, num_prompts, channel, h, w]

                # 扩展权重维度，使其能够与 prompts_tensor 相乘
                weights_expanded = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, Pool_size, 1, 1, 1]
                # 执行加权求和，计算加权后的 prompts
                weighted_prompt = torch.sum(weights_expanded * prompts_tensor, dim=1)  # [batch_size, C, H, W]
                prompts = weighted_prompt

                # # 使用 einsum 直接计算加权和
                # weighted_prompt = torch.einsum('bpchw,bp->bchw', prompts_tensor, weights)  # [B, C, H, W]
                # prompts = weighted_prompt

                # 使用index计算
                # index = res['prompt_idx']
                # prompts_tensor = torch.stack(prompts_list, dim=0)  # [num_prompts, B, channel, h, w]
                # # 转置 prompts_tensor 为 [B, num_prompts, channel, h, w]，方便按 batch 操作
                # prompts_tensor = prompts_tensor.permute(1, 0, 2, 3, 4)  # [B, num_prompts, channel, h, w]

                # # 创建一个列表存储结果
                # selected_prompts_list = []

                # # 遍历每个样本
                # for b in range(index.size(0)):  # 遍历 batch
                #     # 直接通过索引从 prompts_tensor 提取该 batch 的 top_k prompts
                #     selected_prompts = prompts_tensor[b, index[b]]  # [top_k, channel, h, w]
                #     selected_prompts_list.append(selected_prompts)

                # # 将所有样本的 prompts 堆叠为最终的张量
                # selected_prompts = torch.stack(selected_prompts_list, dim=0)  # [B, top_k, channel, h, w]
               
                # selected_prompts = selected_prompts.permute(1, 0, 2, 3, 4)  # [top_k, B, 3, 224, 224]
                # similarities_list = [self.cosine_similarity(x, prompt) for prompt in selected_prompts]  # 每个元素为 [batch_size, 1, h, w]
                # similarities = torch.cat(similarities_list, dim=1)  # 拼接后形状为 [batch_size, top_k, h, w]
                # similarities = similarities * self.args.temperature
                # weights = F.softmax(similarities, dim=1)  # 权重形状为 [batch_size, top_k, h, w]

                # # weight_value = weights[0, 0, 0, 0]  # batch_idx=0, prompt_idx=0, h=0, w=0
                # # print("Weight value:", weight_value.item())  # 转为标量输出

                # selected_prompts = selected_prompts.permute(1, 0, 2, 3, 4)  # [batch_size, top_k, 3, 224, 224]
                # weighted_prompt = torch.sum(weights.unsqueeze(2) * selected_prompts, dim=1)  # 加权求和后形状为 [batch_size, channels, h, w]
                # prompts = weighted_prompt

            else:
                # 使用point-wise的卷积，提升channel维度
                # 特征归一化，例如 BatchNorm 或 GroupNorm，以减少冗余信息
                if self.FFN_input_30prompts:
                    # print("哈哈")
                    processed_input = self.ffn_input(x)
                    # 处理 prompts_list 中的每个 prompt
                    processed_prompts_list = [self.ffn_30prompts(prompt) for prompt in prompts_list]
                    # 对分支进行softmax
                    similarities_list = [self.cosine_similarity(processed_input, prompt) for prompt in processed_prompts_list]  # 每个元素为 [batch_size, 1, h, w]
                else:
                    # 对分支进行softmax
                    similarities_list = [self.cosine_similarity(x, prompt) for prompt in prompts_list]  # 每个元素为 [batch_size, 1, h, w]

                # 拼接所有相似度并通过 softmax 归一化
                similarities = torch.cat(similarities_list, dim=1)  # [batch_size, 20, h, w]
                # print(similarities.shape)
                # similarities = similarities * self.args.temperature
                weights = F.softmax(similarities, dim=1)  # [batch_size, 20, h, w]
                
                # if session == 7 or session == 8 or session == 9:
                #     # Sample 49: Max = 0.18, Min = 0.01
                    
                #     # 遍历第一个维度 (shape[0])，对于每个样本
                #     for i in range(weights.shape[0]):
                #         # 获取每个样本的最大值和最小值，并保留两位小数
                #         max_val = round(weights[i].max().item(), 2)
                #         min_val = round(weights[i].min().item(), 2)
                #         # 输出最大值和最小值
                #         print(f"Sample {i}: Max = {max_val}, Min = {min_val}")
                # 堆叠 prompts 并进行加权求和
                prompts = torch.stack(prompts_list, dim=1)  # [batch_size, 10, channels, h, w]
                weighted_prompt = torch.sum(weights.unsqueeze(2) * prompts, dim=1)  # [batch_size, channels, h, w]
                prompts = weighted_prompt


                # # 对分支进行平均
                # # 堆叠 prompts，得到形状 [batch_size, num_prompts, channels, h, w]
                # prompts = torch.stack(prompts_list, dim=1)  # [batch_size, num_prompts, channels, h, w]
                # # 直接对 num_prompts 维度取均值
                # weighted_prompt = prompts.mean(dim=1)  # [batch_size, channels, h, w]
                # # 更新 prompts
                # prompts = weighted_prompt

        res['prompts'] = prompts
        return res
    
    # （0）保存反归一化之后的原始图片
    def save_batch_as_images(self, input_tensor, folder_path="output_images", file_prefix="image"):
        import os
        
        # 确保文件夹存在
        os.makedirs(folder_path, exist_ok=True)
        
        # 检查输入张量维度
        if input_tensor.dim() != 4 or input_tensor.size(1) != 3:
            raise ValueError("输入张量的维度必须为 [Batch_size, 3, h, w]")
        
        # 将张量裁剪到 [0, 1] 范围
        input_clamped = input_tensor.clip(0, 1)
        
        # 遍历 Batch 中的每一张图片
        batch_size = input_tensor.size(0)
        for i in range(batch_size):
            # 提取单张图片 [3, h, w] -> [h, w, 3]
            image = input_clamped[i].permute(1, 2, 0).cpu().numpy()
            
            # 保存为图片文件
            file_path = os.path.join(folder_path, f"{file_prefix}_{i}.png")
            plt.imsave(file_path, image)
            print(f"Saved: {file_path}")
    
    # （1）保存10个mask
    def save_ring_masks_as_images(self, ring_masks, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_rings = ring_masks.shape[0]
        
        for i in range(num_rings):
            # 获取第 i 个掩码
            mask = ring_masks[i].detach().cpu().numpy()  # 转换为 NumPy 数组

            # 绘制图像
            plt.imshow(mask, cmap='gray')
            plt.axis('off')  # 不显示坐标轴

            # 保存图像
            plt.savefig(os.path.join(save_dir, f'ring_mask_{i}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()  # 关闭图像，以便不与下一个重叠
    
    # （2）保存加权后得到的整个mask
    def save_weighted_mask(self, weighted_ring_masks, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 确保掩码在 CPU 并转换为 NumPy 格式
        weighted_mask = weighted_ring_masks.detach().cpu().numpy()

        # 绘制并保存总掩码
        plt.imshow(weighted_mask, cmap='gray')
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(os.path.join(save_dir, 'weighted_mask.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

    # （3）保存整个mask加到频率域上的图片
    def save_frequency_domain_image(self, fft_selected, save_dir, filename="frequency_domain.png"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 计算频率域的模值 (magnitude)
        magnitude = torch.abs(fft_selected).mean(dim=1)  # 对通道维度取平均，形状为 [Batch_size, h, w]
        magnitude = magnitude.detach().cpu().numpy()  # 转换为 NumPy 格式

        # 归一化到 [0, 1] 范围
        magnitude_normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

        # 保存每个样本的频率域图像
        for i in range(magnitude_normalized.shape[0]):
            plt.imshow(magnitude_normalized[i], cmap='gray')
            plt.axis('off')  # 不显示坐标轴
            plt.savefig(os.path.join(save_dir, f"{filename}_sample_{i}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

    # （4）保存返回到空间区域的图片
    def save_spatial_domain_images(self, ifft_selected, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 转换为 NumPy 格式
        images = ifft_selected.detach().cpu().numpy()  # 转为 NumPy 格式，形状为 [Batch_size, 3, h, w]

        # 对每张图片保存
        for i in range(images.shape[0]):
            image = images[i].transpose(1, 2, 0)  # 转换为 [h, w, 3] 格式
            image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]

            # 保存图片
            plt.imshow(image)
            plt.axis('off')  # 不显示坐标轴
            plt.savefig(os.path.join(save_dir, f"spatial_image_{i}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

    def get_Frequency_mask(self, input):

        # # 保存进行反归一化之后图片
        # save_dir = "/home/jyw/SuperMan/PriViLege_Clear/models/Ouptput/images"
        # self.save_batch_as_images(input, save_dir)
        # sys.exit()

        # 对 h 和 w 维度做傅里叶变换
        fft_im = torch.fft.fftn(input, dim=(-2, -1))  # 2D 傅里叶变换
        fft_im_center = torch.fft.fftshift(fft_im, dim=(-2, -1))  # 将频谱零频率移到中心

        # 构建网格以计算每个点到频谱中心的距离
        Batch_size, channels, h, w = input.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        center_y, center_x = h // 2, w // 2  # 频谱中心
        distances = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)  # 距离矩阵
        distances = distances.to(input.device)  # 确保设备一致

        # 创建圆环掩码，允许一定的容差范围
        beta = 4.0
        ring_masks = []  # 存储每个圆环的掩码
        for i, radius in enumerate(self.radii):
            if i == 0:
                inner_radius = 0  # 第一个圆环从中心开始
            else:
                inner_radius = self.radii[i - 1] + 1e-6  # 确保不重叠

            # 外半径掩码
            outer_mask = torch.sigmoid(-beta * (distances - radius))
            # 内半径掩码
            inner_mask = torch.sigmoid(-beta * (distances - inner_radius))
            # 圆环掩码
            ring_mask = outer_mask - inner_mask
            ring_masks.append(ring_mask.float())  # 转为浮点数，方便后续操作

        # 将掩码堆叠为 [10, h, w] 的张量
        ring_masks = torch.stack(ring_masks, dim=0).to(input.device)  # [10, h, w]

        # # 保存获得的10个mask
        # save_dir = "/home/jyw/SuperMan/PriViLege_Clear/models/Ouptput/ring_masks"
        # self.save_ring_masks_as_images(ring_masks, save_dir)
        # sys.exit()

        # 使用权重对每个圆环加权，此时为了跑最牛性能，把温度给暂时去除
        # weights_normalized = torch.softmax(self.weights, dim=0)  # 权重归一化
     
        weights_normalized = torch.softmax(self.weights * self.args.temperature, dim=0)  # 权重归一化
        weighted_ring_masks = weights_normalized[:, None, None] * ring_masks  # 权重加权的掩码
        # 对加权掩码求和，得到整体频率掩码
        final_mask = weighted_ring_masks.sum(dim=0)  # [h, w]

        # # 保存获得的10个mask
        # save_dir = "/home/jyw/SuperMan/PriViLege_Clear/models/Ouptput/ring_masks_weighted"
        # self.save_weighted_mask(final_mask, save_dir)
        # sys.exit() 

        # 应用频率掩码
        fft_selected = fft_im_center * final_mask[None, None, :, :]  # 广播到 [Batch_size, 3, h, w]

        # save_dir = "/home/jyw/SuperMan/PriViLege_Clear/models/Ouptput/fft_selected_firstbatch"
        # self.save_frequency_domain_image(fft_selected, save_dir)
        # sys.exit()  # 只保留第一个batch的

        # （3）运用残差操作
        fft_residual = fft_im_center + fft_selected  # 原始频率 + 加权圆环
        ifft_residual = torch.fft.ifftn(torch.fft.ifftshift(fft_residual, dim=(-2, -1)), dim=(-2, -1))
        ifft_residual = torch.abs(ifft_residual)  # [Batch_size, 3, h, w]
        output = input + (ifft_residual - input) * 0.1
        # 多了一步，把圆环加到input频率域上，返回到空间域，再减去input。获得学习到的信息。
        # 来源于，我想把圆环加到input的频率域上，直接作为接下来的input。
        return output

        # # （2）直接在频率域操作
        # fft_combined = fft_im_center + fft_selected  # 原始频率图像与掩码加权的频率圆环叠加
        # ifft_combined = torch.fft.ifftn(torch.fft.ifftshift(fft_combined, dim=(-2, -1)), dim=(-2, -1))
        # ifft_combined = torch.abs(ifft_combined)  # [Batch_size, 3, h, w]
        # return ifft_combined
    

        # # （1）返回到时域
        # ifft_selected = torch.fft.ifftn(torch.fft.ifftshift(fft_selected, dim=(-2, -1)), dim=(-2, -1))
        # ifft_selected = torch.abs(ifft_selected)  # [Batch_size, 3, h, w]

        # # save_dir = "/home/jyw/SuperMan/PriViLege_Clear/models/Ouptput/ifft_selected_firstbatch"
        # # self.save_spatial_domain_images(ifft_selected, save_dir)
        # # sys.exit()  # 只保留第一个batch的

        # return ifft_selected

    def get_Frequency_mask_patch(self, input):
        # 对 h 和 w 维度做傅里叶变换
        fft_im = torch.fft.fftn(input, dim=(-2, -1))  # 2D 傅里叶变换
        fft_im_center = torch.fft.fftshift(fft_im, dim=(-2, -1))  # 将频谱零频率移到中心

        # 图像的大小和 Patch 尺寸
        _, _, h, w = input.shape
        P = 16  # 每个 Patch 的大小

        # 创建 Patch 掩码，初始为 zeros
        patch_masks = torch.zeros((196, h, w), device=input.device)

        # 生成 Patch 坐标
        patch_idx = 0
        for i in range(0, h, P):
            for j in range(0, w, P):
                patch_masks[patch_idx, i:i+P, j:j+P] = 1.0
                patch_idx += 1

        # 使用权重对每个 Patch 加权
        weights_normalized = torch.softmax(self.weights * self.args.temperature, dim=0)  # 权重归一化
        weighted_patches = weights_normalized[:, None, None] * patch_masks  # 权重加权的 Patch 掩码
        final_mask = weighted_patches.sum(dim=0)  # 合成最终掩码，大小为 [h, w]

        # 应用频率掩码
        fft_selected = fft_im_center * final_mask[None, None, :, :]  # 广播到 [Batch_size, 3, h, w]

        # 返回到时域
        ifft_selected = torch.fft.ifftn(torch.fft.ifftshift(fft_selected, dim=(-2, -1)), dim=(-2, -1))
        ifft_selected = torch.abs(ifft_selected)  # [Batch_size, 3, h, w]

        return ifft_selected

    def get_Frequency_mask_AdaptiveFrequencyMask(self, input, session):
        # 对 h 和 w 维度做傅里叶变换
        fft_im = torch.fft.fftn(input, dim=(-2, -1))  # 2D 傅里叶变换
        fft_im_center = torch.fft.fftshift(fft_im, dim=(-2, -1))  # 将频谱零频率移到中心

        final_mask, control_points, weights = self.AdaptiveFrequencyMask(input)
        if(session == 10):
            print(control_points)
            print(weights)

        if(session == 10):
            # 保存加权后得到的整个mask
            save_dir = "/home/jyw/SuperMan/PriViLege_Clear/models/Ouptput/OneMask"
            self.save_weighted_mask(final_mask, save_dir)
            # sys.exit() 


        # 应用频率掩码
        fft_selected = fft_im_center * final_mask[None, None, :, :]  # 广播到 [Batch_size, 3, h, w]

        if(session == 10):
            save_dir = "/home/jyw/SuperMan/PriViLege_Clear/models/Ouptput/AfterMask"
            self.save_frequency_domain_image(fft_selected, save_dir)
            # sys.exit()  # 只保留第一个batch的

        # 返回到时域
        ifft_selected = torch.fft.ifftn(torch.fft.ifftshift(fft_selected, dim=(-2, -1)), dim=(-2, -1))
        ifft_selected = torch.abs(ifft_selected)  # [Batch_size, 3, h, w]

        if(session == 10):
            save_dir = "/home/jyw/SuperMan/PriViLege_Clear/models/Ouptput/Final"
            self.save_spatial_domain_images(ifft_selected, save_dir)
            # sys.exit()  # 只保留第一个batch的

        return ifft_selected

    def prompt_output(self, before, after, prompt, epoch, batch_num):
        denormalize_transform = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            transforms.ToPILImage()
        ])
        batch_size = before.size(0)

        # os.makedirs(before_path, exist_ok=True)
        before_path = "/home/jyw/SuperMan/PriViLege_InsVP/prompt_output/before"
        os.makedirs(before_path, exist_ok=True)
        for i in range(batch_size):
            # 对每张图片应用反归一化 transform
            image = denormalize_transform(before[i].cpu())
            # 保存图片
            image.save(os.path.join(before_path, f"epoch{epoch}_batch{batch_num}_{i}.png"))
        
        # os.makedirs(after_path, exist_ok=True)
        after_path = "/home/jyw/SuperMan/PriViLege_InsVP/prompt_output/after"
        os.makedirs(after_path, exist_ok=True)
        for i in range(batch_size):
            # 对每张图片应用反归一化 transform
            image = denormalize_transform(after[i].cpu())
            # 保存图片
            image.save(os.path.join(after_path, f"epoch{epoch}_batch{batch_num}_{i}.png"))

        # os.makedirs(prompt_path, exist_ok=True)
        prompt_path = "/home/jyw/SuperMan/PriViLege_InsVP/prompt_output/prompt"
        os.makedirs(prompt_path, exist_ok=True)
        for i in range(batch_size):
            # 对每张图片应用反归一化 transform
            image = denormalize_transform(prompt[i].cpu())
            # 保存图片
            image.save(os.path.join(prompt_path, f"epoch{epoch}_batch{batch_num}_{i}.png"))

    def forward(self, input, prompt_feat=False, B_tuning=False, base=False, query=False, eval=False, memory_data=None, session=-1, Mytest=False):

        res = {}  # 初始化 res 为一个空字典
        
        if self.args.test1: # 加权
            if self.args.pixel_prompt == 'YES':  # 使用卷积核拿到
                res = self.get_prompts(input, session=session)  
                prompts = res['prompts']
                input1 = input + prompts * 1
            if self.args.Frequency_mask:
                input2 = self.get_Frequency_mask(input)
            input = self.alpha * input1 + self.beta * input2
        else: # 前后
            if self.args.pixel_prompt == 'YES':  # 使用卷积核拿到
                res = self.get_prompts(input, session=session)  
                prompts = res['prompts']
                input = input + prompts * 1
            if self.args.Frequency_mask:
                input = self.get_Frequency_mask(input)
            
            # print(self.weights)
            # print("哈哈")

        if base:
            embedding = self.prompt_encode(input, prompt_feat=True, B_tuning=True, eval=eval)
            cls_embed, prompt_embed = embedding
            # fc原来在这里哈哈哈哈
            logit = self.fc(0.5*(prompt_embed['Vision']+cls_embed))
            return logit, cls_embed, prompt_embed
        if query: # 在更新分类器权重的时候，这里直接没有使用prompt
            q_feat = self.encode(input)
            return q_feat

        if Mytest: # 此时test直接通过标准的vit
            q_feat = self.encode(input)
            logit = self.fc(q_feat)
            res['logit'] = logit
            return res

        embedding = self.prompt_encode(input, prompt_feat=prompt_feat, B_tuning=B_tuning, eval=eval)
        logit = self.fc(embedding)

        res['logit'] = logit

        if memory_data is not None:  # 把通过高斯分布得到数据拿到
            logit = torch.cat([logit, memory_data], dim=0)
        return res

    def train_inc(self, dataloader, epochs, session, class_list, testloader, result_list, test, model_test):
        print("[Session: {}]".format(session))
        self.update_fc_avg(dataloader, class_list)
        optimizer_params = []

        if self.args.Frequency_mask: # 在novel阶段也优化权重
            params_Frequency_mask = [self.weights]
            optimizer_params.append({'params': params_Frequency_mask, 'lr': self.args.lr_Frequency_mask * 0.05})

        # VPT
        params_vpt = [self.Prompt_Tokens]
        optimizer_params.append({'params': params_vpt, 'lr': self.args.lr_PromptTokens_novel})

        # 分类器
        params_classsifier = [p for p in self.fc.parameters()]
        optimizer_params.append({'params': params_classsifier, 'lr': self.args.lr_new})

        optim = torch.optim.Adam(optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * 1)
        
        best_epoch = -1
        best_accuracy = 0.0

        for epoch in range(epochs):
            # 利用高斯分布生成Replay样本
            if self.args.RAPF == 'YES':
                random_class_order_list = list(range(self.base_class + (session - 1) * self.way)) # 注意只是前面的增量阶段
                random.shuffle(random_class_order_list)

            batch_id = -1
            for idx,batch in enumerate(dataloader):
                data_imgs, data_label = [_.cuda() for _ in batch]

                if self.args.RAPF == 'YES': # 使用高斯重放
                    batch_id += 1
                    sg_inputs = []
                    sg_targets = []
                    list_for_one_batch = [random_class_order_list[batch_id*2%len(random_class_order_list)], random_class_order_list[(batch_id*2+1)%len(random_class_order_list)]]
                    for i in list_for_one_batch:
                            # 在cub数据集中，每个session会有 10*5个样本，这里我生成了1/3个
                            sg_inputs.append(sample(self.class_mean_list[i], self.class_cov_list[i], int(10), shrink=True))
                            sg_targets.append(torch.ones(int(10), dtype=torch.long, device=self.device)*i) # label默认是这个类对应的id
                    sg_inputs = torch.cat(sg_inputs, dim=0)
                    sg_targets = torch.cat(sg_targets, dim=0)
                    data_label = torch.cat([data_label, sg_targets], dim=0)  # 赋值给label

                if self.args.RAPF == 'NO':  
                    sg_inputs=None
        
                self.train()

                res = self.forward(data_imgs, memory_data=sg_inputs, session=session)
                logits = res['logit']

                seen_class = self.base_class + session * self.way
                logits = logits[:, :seen_class] #跟 test是一样的

                loss_ce = F.cross_entropy(logits, data_label)
                loss = loss_ce
                
                optim.zero_grad()
                loss.backward()
                
                optim.step()
                scheduler.step()
                pred = torch.argmax(logits, dim=1)
                acc = (pred == data_label).sum().item()/data_label.shape[0]*100.
                
            lrc = scheduler.get_last_lr()[0]  # 得到当前的学习率
            tsl, tsa, logs = test(model_test, testloader, 0, self.args, session, Mytest=False)
            if tsa > best_accuracy:
                best_accuracy = tsa
                best_epoch = epoch

            result_list.append(
                        'epoch:%03d,lr:%.4f,B:%.5f,N:%.5f,BN:%.5f,NB:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, logs['base_acc'], logs['new_acc'], logs['base_acc_given_new'], logs['new_acc_given_base'], loss, acc, tsl, tsa
                        )
                    )
        result_list.append('Session {}, Best test_Epoch {}, Best test_Acc {:.4f}'.format(
                    session, best_epoch, best_accuracy))

        return tsa
    
    def update_fc_avg(self,dataloader,class_list):
        self.eval()
        query_p=[]
        
        with torch.no_grad():
            for batch in dataloader:
                data_imgs, label = [_.cuda() for _ in batch]
                cls_embed=self.encode(data_imgs).detach()
            
            for class_index in class_list:
                data_index=(label==class_index).nonzero().squeeze(-1)
                embedding = cls_embed[data_index]
                proto=embedding.mean(0)
                query_p.append(proto)
                self.fc.weight.data[class_index]=proto
            query_p = torch.stack(query_p)
        # query_info["proto"] = torch.cat([query_info["proto"], query_p.cpu()])
        
        self.train()

    # 对均值和方差进行纠正
    def analyze_mean_cov(self, features, labels):
        label = torch.sort(torch.unique(labels))[0]
        for l in label:
            index = torch.nonzero(labels == l)
            index = index.squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            cov = torch.cov(class_data.t()) + 1e-4* torch.eye(class_data.shape[-1], device=class_data.device)
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)