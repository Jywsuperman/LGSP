import os
import sys

# Random seeds for reproducible experiments
# seeds = [1,2,3,4,5]
seeds = [1]

# Basic configuration
project = "base"
dataset = "cub200"
# dataset = 'FGVCAircraft'
# dataset = 'iNF200'
gpu_num = 1

# Model hyperparameters
lr_Frequency_mask = 0.1
temperature = 2

# Learning rates and dropout for 30-branch network
lr_InsVP = 0.01
Dropout_Prompt = 0.3

# Learning rates and dropout for 9-12 layer 1D convolution
lr_Block = 0.1
Dropout_Block = 0.3

# Training epochs
epochs_bases = [80]
epochs_new = 5

# Learning rates for prompt tokens (base and novel sessions)
lr_PromptTokens_base = 0.02
lr_PromptTokens_novel = 0.003

# Learning rates for classifier (base and novel sessions)
lr_base = 0.01
lr_new = 0.06

# Alternative settings for quick testing
# epochs_bases = [1]
# epochs_new = 1

# Training schedule
milestones_list = ["20"]
Prompt_Token_num = 5
lr_prompt_l2p = 0.002

# Baseline configuration (commented out)
# epochs_bases = [19]
# epochs_new = 5
# lr_PromptTokens_base = 0.01
# lr_PromptTokens_novel = 0.001
# lr_base = 0.012
# lr_new = 0.03

# Data directory path
data_dir = "/path/to/your_workspace/data"

# Training loop
for seed in seeds:
    print("Pretraining -- Seed{}".format(seed))
    for i, epochs_base in enumerate(epochs_bases):
        os.system(
            ""
            "python train.py "
            "-project {} "
            "-dataset {} "
            "-base_mode ft_dot "
            "-new_mode avg_cos "
            "-gamma 0.1 "
            "-lr_base {} "
            "-lr_new {} "
            "-lr_InsVP {} "
            "-decay 0.0005 "
            "-epochs_base {} "
            "-epochs_new {} "
            "-schedule Cosine "
            "-milestones {} "
            "-gpu {} "
            "-temperature 16 "
            "-start_session 0 "
            # '-batch_size_base 128 '
            "-batch_size_base 64 "
            "-seed {} "
            "-vit "
            # '-clip'
            "-comp_out 1 "
            "-prefix "
            # '-ED '
            # '-SKD '
            "-LT "
            "-out {} "
            "-Prompt_Token_num {} "
            "-lr_PromptTokens_base {} "
            "-lr_PromptTokens_novel {} "
            "-lr_Block {} "
            "-Dropout_Block {} "
            "-Dropout_Prompt {} "
            "-temperature {} "
            "-lr_prompt_l2p {} "
            "-lr_Frequency_mask {} "
            "-dataroot {}".format(project, dataset, lr_base, lr_new, lr_InsVP, epochs_base, epochs_new, milestones_list[i], gpu_num, seed, "PriViLege", Prompt_Token_num, lr_PromptTokens_base, lr_PromptTokens_novel, lr_Block, Dropout_Block, Dropout_Prompt, temperature, lr_prompt_l2p, lr_Frequency_mask, data_dir)
        )

# Note:
# -prefix: If enabled, uses prompt to concatenate intermediate layer features
# If disabled, uses the proposed method to concatenate MLP and MSA features
# in the first two layers (main contribution of this paper)
