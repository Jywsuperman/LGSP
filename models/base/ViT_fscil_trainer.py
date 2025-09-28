from .base import Trainer
import os.path as osp
import torch.nn as nn
from .parallel import DataParallelModel, DataParallelCriterion
import copy
from copy import deepcopy
import pandas as pd
from os.path import exists as is_exists
import time

from .helper import *
from utils import *
from dataloader.data_utils import *
from models.switch_module import switch_module
from dataloader.data_manager import DataManager
# from models.prompt import Global_Prompt_Extractor
import sys

from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/all_phases')

class ViT_FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.set_log_path()

        self.args = set_up_datasets(self.args)
        self.model = ViT_MYNET(self.args, mode=self.args.base_mode)
        self.model = self.model.cuda()
        
        for p in self.model.encoder.parameters():
            p.requires_grad=False
        
        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            # self.best_model_dict = deepcopy(self.model.state_dict())
    
        print("#"*50)
        trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        self.init_params = sum(param.numel() for param in self.model.parameters())
        print('total parameters:',self.init_params)
        print('trainable parameters:',trainable_params)
        print("#"*50)
        # self.writer = SummaryWriter(f'runs/all_phases_val/epoch{self.args.epochs_base}_{self.args.epochs_new}')

    def get_optimizer_base(self):
        optimizer_params = []

        
        if self.args.pixel_prompt == "YES":
            params_Mask = [
            p for p in 
            list(self.model.meta_net_3.parameters()) + list(self.model.meta_net_2.parameters()) + 
            [param for i in range(29) for param in getattr(self.model, f"First_Pool_Prompt_Net{i}").parameters()]
            if p.requires_grad]
            optimizer_params.append({'params': params_Mask, 'lr': self.args.lr_InsVP})

        if self.args.Block_prompt == "YES":
            params_Block_9_12 = [p for p in list(self.model.meta_net_block_0.parameters()) + list(self.model.meta_net_block_1.parameters()) + list(self.model.meta_net_block_2.parameters()) + list(self.model.meta_net_block_3.parameters()) if p.requires_grad]
            optimizer_params.append({'params': params_Block_9_12, 'lr': self.args.lr_Block})
        
        if self.args.prompt_pool == True:
            params_l2p_prompt = [p for p in self.model.prompt_l2p.parameters() if p.requires_grad]
            optimizer_params.append({'params': params_l2p_prompt, 'lr': self.args.lr_prompt_l2p})

  
        if self.args.Frequency_mask:
            params_Frequency_mask = [self.model.weights]
            optimizer_params.append({'params': params_Frequency_mask, 'lr': self.args.lr_Frequency_mask})
        
        if self.args.FFN_input_30prompts:
            params_FFN_input_30prompts = [
                p for p in list(self.model.ffn_input.parameters()) + list(self.model.ffn_30prompts.parameters()) 
                if p.requires_grad
            ]
            optimizer_params.append({'params': params_FFN_input_30prompts, 'lr': self.args.lr_FFN_input_30prompts})


        if self.args.test1:
            params_test1 = [self.model.alpha, self.model.beta]
            optimizer_params.append({'params': params_test1, 'lr': 0.1})
        # VPT
        params_vpt = [self.model.Prompt_Tokens]
        optimizer_params.append({'params': params_vpt, 'lr': self.args.lr_PromptTokens_base})

  
        params_classsifier = [p for p in self.model.fc.parameters()]
        optimizer_params.append({'params': params_classsifier, 'lr': self.args.lr_base})

        optimizer = torch.optim.Adam(optimizer_params)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()
        result_list = [args]
        columns = ['num_session', 'acc', 'base_acc', 'new_acc', 'base_acc_given_new', 'new_acc_given_base']
        acc_df = pd.DataFrame(columns=columns)
        print("[Start Session: {}] [Sessions: {}]".format(args.start_session, args.sessions))
        
        for session in range(args.start_session, args.sessions):
            
            train_set, trainloader, testloader = self.get_dataloader(session)
            print(f"Session: {session} Data Config")
            print(len(train_set.targets))
            if session == 0:
                for p in self.model.meta_net_3.parameters():
                    p.requires_grad = False

                if self.args.pixel_prompt == "YES":
                    checkpoint_path = "./local_datasets//run_script/meta_net_2_params_lastBaseEpoch.pth"
                    state_dict = torch.load(checkpoint_path)

                  
                    self.model.meta_net_2.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net0.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net1.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net2.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net3.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net4.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net5.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net6.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net7.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net8.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net9.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net10.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net11.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net12.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net13.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net14.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net15.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net16.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net17.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net18.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net19.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net20.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net21.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net22.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net23.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net24.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net25.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net26.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net27.load_state_dict(state_dict)
                    self.model.First_Pool_Prompt_Net28.load_state_dict(state_dict)
                    

                if self.args.Block_prompt == "YES":
                    checkpoint_path = "./local_datasets//run_script/meta_net_3_params_lastBaseEpoch.pth"
                    state_dict = torch.load(checkpoint_path)
                    self.model.meta_net_block_0.load_state_dict(state_dict)
                    self.model.meta_net_block_1.load_state_dict(state_dict)
                    self.model.meta_net_block_2.load_state_dict(state_dict)
                    self.model.meta_net_block_3.load_state_dict(state_dict)
                  
                
            if session > 0: 
                
                for p in self.model.meta_net_3.parameters():
                    p.requires_grad = False
                
                if self.args.pixel_prompt == "YES":
                    for p in self.model.meta_net_2.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net0.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net1.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net2.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net3.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net4.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net5.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net6.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net7.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net8.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net9.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net10.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net11.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net12.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net13.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net14.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net15.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net16.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net17.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net18.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net19.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net20.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net21.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net22.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net23.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net24.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net25.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net26.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net27.parameters():
                        p.requires_grad = False
                    for p in self.model.First_Pool_Prompt_Net28.parameters():
                        p.requires_grad = False



                
                if self.args.Block_prompt == "YES":
                    for p in self.model.meta_net_block_0.parameters():
                        p.requires_grad = False
                    for p in self.model.meta_net_block_1.parameters():
                        p.requires_grad = False
                    for p in self.model.meta_net_block_2.parameters():
                        p.requires_grad = False
                    for p in self.model.meta_net_block_3.parameters():
                        p.requires_grad = False

                if self.args.prompt_pool:
                    
                    for p in self.model.prompt_l2p.parameters():
                        p.requires_grad = False
                 
                if self.args.Frequency_mask:
                    self.model.weights.requires_grad = True
                    # for p in self.model.AdaptiveFrequencyMask.parameters():
                    #     p.requires_grad = False

                if self.args.FFN_input_30prompts:
                    for p in list(self.model.ffn_input.parameters()) + list(self.model.ffn_30prompts.parameters()): 
                        p.requires_grad = False
                
               

            #todo ===============================================
            if session == 0:  # load base class train img label
                
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                
                # build_base_proto(trainloader, self.model, self.query_info, args)
                trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
                print('[Session {}] Trainable parameters: {}'.format(session,trainable_params))
                print("#"*50)
            
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, np.unique(train_set.targets), args)
                    tsl, tsa, logs = test(self.model, testloader, epoch, args, session, Mytest=False)

                    # self.writer.add_scalar('Accuracy/Base_Phase', tsa, epoch)

                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'][session] = epoch
                      
                        print('********A better model is found!!**********')
                        # print('Saving model to :%s' % save_model_dir)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,B:%.5f,N:%.5f,BN:%.5f,NB:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, logs['base_acc'], logs['new_acc'], logs['base_acc_given_new'], logs['new_acc_given_base'], tl, ta*100, tsl, tsa*100
                        )
                    )
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                            '\nstill need around %.2f mins to finish this session' % (
                                    (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'][session], self.trlog['max_acc'][session], ))

                #*=======================================================================================
                if not args.not_data_init:
                    # if not args.pret_clip:
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    self.model.mode = 'avg_cos'
                    tsl, tsa, logs = test(self.model, testloader, 0, args, session, Mytest=False)
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    result_list.append('After Prototype FC: test_loss:%.5f,test_acc:%.5f\n' % (tsl, tsa))

            else:  # incremental learning sessions
                print("Incremental session: [%d]" % session)
                print("#"*50)
                trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
                print('[Session {}] Trainable parameters: {}'.format(session,trainable_params))
                print("#"*50)
            
                self.model.update_seen_classes(np.unique(train_set.targets))

                self.model.mode = self.args.new_mode
                self.model.train()
                trainloader.dataset.transform = testloader.dataset.transform
                # self.model.module.train_inc(trainloader, self.args.epochs_new, session, np.unique(train_set.targets), self.word_info, self.query_info)
                tsa = self.model.train_inc(trainloader, self.args.epochs_new, session, np.unique(train_set.targets), testloader, result_list, test, self.model)
                
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))
                
            if self.args.RAPF == 'YES':
               
                sample_loader = get_dataloader_RAPF(self.args, session)
                sample_data = []
                sample_target = []
                with torch.no_grad():
                    tqdm_gen = tqdm(sample_loader)
                    for i, batch in enumerate(tqdm_gen, 1):
                        data, label = [_.cuda() for _ in batch]
                        logits = self.model(data)
                        sample_data.append(logits)
                        sample_target.append(label)
                    sample_target = torch.cat(sample_target, dim=0)
                    sample_data = torch.cat(sample_data, dim=0)
                    self.model.analyze_mean_cov(sample_data, sample_target)

        result_list.append(self.trlog['max_acc'])

        print(self.trlog['max_acc'])

      
        print()
        max_acc = self.trlog['max_acc']
        first_value = max_acc[0]
        last_value = max_acc[-1]
        average = sum(max_acc) / len(max_acc)
        print(f"{first_value:.3f}  {last_value:.3f}  {average:.3f}")
        
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'][0])
        print('Total time used %.2f mins' % total_time)
        
        end_params = sum(param.numel() for param in self.model.parameters())
        print('[Begin] Total parameters: {}'.format(self.init_params))
        print('[END] Total parameters: {}'.format(end_params))
        
    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/%s/' % (self.args.dataset, time.strftime("%Y%m%d_%H%M%S"))
        if self.args.vit:
            self.args.save_path = self.args.save_path + '%s/' % (self.args.project+'_ViT_Ours')
        else:
            self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.decay, self.args.seed)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.decay, self.args.seed)
        else:
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-COS_%d-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.decay, self.args.seed)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join(f'checkpoint/{self.args.out}', self.args.save_path)
        ensure_path(self.args.save_path)
        return None

    def set_log_path(self):
        if self.args.model_dir is not None:
            self.args.save_log_path = '%s/' % self.args.project
            self.args.save_log_path = self.args.save_log_path + '%s' % self.args.dataset
            if 'avg' in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + '_prototype_' + self.args.model_dir.split('/')[-2][:7] + '/'
            if 'ft' in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + '_WaRP_' + 'lr_new_%.3f-epochs_new_%d-keep_frac_%.2f/' % (
                    self.args.lr_new, self.args.epochs_new, self.args.fraction_to_keep)
            self.args.save_log_path = os.path.join('acc_logs', self.args.save_log_path)
            ensure_path(self.args.save_log_path)
            self.args.save_log_path = self.args.save_log_path + self.args.model_dir.split('/')[-2] + '.csv'

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)