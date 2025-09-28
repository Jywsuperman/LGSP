import os
import sys
# seeds = [1,2,3,4,5]
seeds = [1]

project = 'base'
dataset = 'cub200'
# dataset = 'FGVCAircraft'
# dataset = 'iNF200'
gpu_num = 1

lr_Frequency_mask = 0.1

temperature = 2

# 30分支的学习率和dropout
lr_InsVP = 0.01
Dropout_Prompt = 0.3

# 9-12层一维卷积的学习率和dropout
lr_Block = 0.1
Dropout_Block = 0.3

epochs_bases = [80]
epochs_new = 5

# prompt的base和novel学习率
lr_PromptTokens_base = 0.02
lr_PromptTokens_novel = 0.003

# 分类器的base和novel学习率
lr_base = 0.01
lr_new = 0.06

# epochs_bases = [1]
# epochs_new = 1
milestones_list = ['20']
Prompt_Token_num = 5

lr_prompt_l2p = 0.002
# Baseline指标
# epochs_bases = [19]
# epochs_new = 5

# # prompt的base和novel学习率
# lr_PromptTokens_base = 0.01
# lr_PromptTokens_novel = 0.001

# # 分类器的base和novel学习率
# lr_base = 0.012
# lr_new = 0.03

data_dir = '/home/jyw/data'

for seed in seeds:
    print("Pretraining -- Seed{}".format(seed))
    for i, epochs_base in enumerate(epochs_bases):
        os.system(''
                'python train.py '
                '-project {} '
                '-dataset {} '
                '-base_mode ft_dot '
                '-new_mode avg_cos '
                '-gamma 0.1 '
                '-lr_base {} '
                '-lr_new {} '
                '-lr_InsVP {} '
                '-decay 0.0005 '
                '-epochs_base {} '
                '-epochs_new {} '
                '-schedule Cosine '
                '-milestones {} '
                '-gpu {} '
                '-temperature 16 '
                '-start_session 0 '
                # '-batch_size_base 128 '
                '-batch_size_base 64 '
                '-seed {} '
                '-vit '
                # '-clip'
                '-comp_out 1 '
                '-prefix '
                # '-ED '
                # '-SKD '
                '-LT '
                '-out {} '
                '-Prompt_Token_num {} '
                '-lr_PromptTokens_base {} '
                '-lr_PromptTokens_novel {} '
                '-lr_Block {} '
                '-Dropout_Block {} '
                '-Dropout_Prompt {} '
                '-temperature {} '
                '-lr_prompt_l2p {} '
                '-lr_Frequency_mask {} '
                '-dataroot {}'.format(project, dataset, lr_base, lr_new, lr_InsVP, epochs_base, epochs_new, milestones_list[i], gpu_num, seed, 'PriViLege', Prompt_Token_num, lr_PromptTokens_base, lr_PromptTokens_novel, lr_Block, Dropout_Block, Dropout_Prompt, temperature, lr_prompt_l2p, lr_Frequency_mask, data_dir)
                )

# prefix：如果存在的话，使用prompt对中间层的特征进行cat
# 如果不存在的话，使用论文提出的在前两层对MLP和MSA的特征进行cat（本文的idea）