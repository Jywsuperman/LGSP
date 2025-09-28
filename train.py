import argparse
import importlib
from utils import *
import torch

MODEL_DIR=None
DATA_DIR = './local_datasets/'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-out', type=str, default=None)

    parser.add_argument('-epochs_base', type=int, default=50)
    parser.add_argument('-epochs_new', type=int, default=20)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=2e-4)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=80)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=64)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=128)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos', 'ft_comb', 'ft_euc']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    # for cec
    parser.add_argument('-lrg', type=float, default=0.1) #lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)

    parser.add_argument('-start_session', type=int, default=1)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='2, 3')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=-1)
    # parser.add_argument('-seeds', nargs='+', help='<Required> Set flag', required=True, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-rotation', action='store_true')
    parser.add_argument('-fraction_to_keep', type=float, default=0.1)
    
    parser.add_argument('-vit', action='store_true')
    parser.add_argument('-baseline', action='store_true')
    parser.add_argument('-clip', action='store_true')
    
    parser.add_argument('-ED', action='store_true')
    parser.add_argument('-ED_hp', type=float, default=0.1)
    
    parser.add_argument('-LT', action='store_true')
    parser.add_argument('-WC', action='store_true')
    parser.add_argument('-MP', action='store_true')
    
    parser.add_argument('-SKD', action='store_true')
    parser.add_argument('-l2p', action='store_true')
    parser.add_argument('-dp', action='store_true')
    parser.add_argument('-prefix', action='store_true')
    parser.add_argument('-pret_clip', action='store_true')
    parser.add_argument('-comp_out', type=int, default=1.)
    
    # parser.add_argument('-scratch', action='store_true')
    parser.add_argument('-scratch', action='store_true')
    parser.add_argument('-taskblock', type=int, default=3)
    parser.add_argument('-ft', action='store_true')
    parser.add_argument('-lp', action='store_true')
    parser.add_argument('-PKT_tune_way', type=int, default=1)

    # for LGSP
    parser.add_argument('-VPT_type', type=str, default='deep', 
                        choices=['shallow', 'deep']) 
    parser.add_argument('-Prompt_Token_num', type=int, default=5) 
    parser.add_argument('-lr_PromptTokens_base', type=float, default=2e-4) 
    parser.add_argument('-lr_PromptTokens_novel', type=float, default=2e-4)

    parser.add_argument('--InsVP_hid_dim', type=int, 
                    default=30,
                    help='The hidden dimension of InstanceVP.')

    parser.add_argument('--InsVP_prompt_patch', type=int,
                        default=16,
                        help='The prompt patch size of InstanceVP.')
    parser.add_argument("--InsVP_prompt_patch_2", type=int,
                        default=3, # 11
                        help="The prompt patch size of InstanceVP.")
    parser.add_argument("--InsVP_prompt_patch_22", type=int, 
                        default=5, # 25
                        help="The prompt patch size of InstanceVP.")
    parser.add_argument("--InsVP_hid_dim_2", type=int,  
                        default=3,
                        help="The hidden dimension of InstanceVP.")
    parser.add_argument('-lr_InsVP', type=float, default=2e-4)  

    parser.add_argument('-lr_Block', type=float, default=2e-4) 
    parser.add_argument('-Dropout_Block', type=float, default=0.9) 
    
    parser.add_argument('-RAPF', type=str, default='NO', 
                        choices=['NO', 'YES']) 

    parser.add_argument('-First_Pool_Prompt_Net0', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-First_Pool_Prompt_Net1', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-First_Pool_Prompt_Net2', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-First_Pool_Prompt_Net3', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-First_Pool_Prompt_Net4', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-First_Pool_Prompt_Net5', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-First_Pool_Prompt_Net6', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-First_Pool_Prompt_Net7', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-First_Pool_Prompt_Net8', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-First_Pool_Prompt_Net9', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-First_Pool_Prompt_Net10_23', type=str, default='YES', 
                        choices=['NO', 'YES']) 
    parser.add_argument('-Dropout_Prompt', type=float, default=0.1) 


    parser.add_argument('-prompt_pool', default=False, type=bool,) 
    parser.add_argument('-pool_size', default=30, type=int,) 
    parser.add_argument('-embedding_key', default='cls', type=str)
    parser.add_argument('-top_k', default=10, type=int, )
    parser.add_argument('-prompt_key_init', default='proto_fc', type=str)
    parser.add_argument('-use_prompt_mask', default=False, type=bool)
    parser.add_argument('-batchwise_prompt', default=False, type=bool)
    parser.add_argument('-pull_constraint', default=True) 
    parser.add_argument('-pull_constraint_coeff', default=0.9, type=float) 
    parser.add_argument('-lr_prompt_l2p', type=float, default=0.03) 

    parser.add_argument('-FFN_input_30prompts', default=False, type=bool,) 
    parser.add_argument('-lr_FFN_input_30prompts', type=float, default=0.02) 

    # VFPT
    parser.add_argument('-VFPT', default=False, type=bool,) # False # True

    parser.add_argument('-VPT', type=str, default='YES', 
                        choices=['NO', 'YES']) 

    parser.add_argument('-pixel_prompt', type=str, default='YES',
                        choices=['NO', 'YES']) 
    parser.add_argument('-Block_prompt', type=str, default='NO',   
                        choices=['NO', 'YES']) 
    parser.add_argument('-Frequency_mask', default=True, type=bool,) # False # True
    parser.add_argument('-lr_Frequency_mask', type=float, default=0.03) 
    parser.add_argument('-num_r', default=80, type=int, ) # FVG

    parser.add_argument('-test1', default=False, type=bool,)

    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100', 'FGVCAircraft', 'iNF200'])

    return parser


if __name__ == '__main__':
    
    parser = get_command_line_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    if args.vit:
        if args.baseline:
            trainer = importlib.import_module('models.%s.ViT_fscil_Baseline_trainer' % (args.project)).ViT_Baseline_FSCILTrainer(args)
        #* L2P / DP
        elif args.l2p:
            pass
        elif args.dp:
            pass
        else:
            trainer = importlib.import_module('models.%s.ViT_fscil_trainer' % (args.project)).ViT_FSCILTrainer(args)
        
    elif args.clip:
        trainer = importlib.import_module('models.%s.CLIP_fscil_trainer' % (args.project)).CLIP_FSCILTrainer(args)
    else:
        trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()