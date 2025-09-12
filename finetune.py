import pandas as pd
import random
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
import math
import re
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import RobertaConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import mean_absolute_error
from functools import partial
from tqdm import tqdm, trange
import copy
import warnings
import oracle
import gc  
warnings.filterwarnings("ignore")

# from roberta_regression import RobertaForRegression, BertForSequenceClassification
from trainer import Trainer, TrainerConfig
from dataset import DNA_reg_Dataset, SimpleDNATokenizer, DNA_reg_conv_Dataset
from Enformer import BaseModel, BaseModelMultiSep, ConvHead, EnformerTrunk, TimedEnformerTrunk
from oracle import DNAQualityAssessor

import wandb 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_from_files(root_path, split, ids):
    dataset = []
    for id in range(ids):
        with open(root_path+'_'+split+'_'+str(id)+'.txt', 'r') as file:
            dataset.extend(file.readlines())
            print('loaded dataset from '+root_path+'_'+split+'_'+str(id)+'.txt')
    return dataset

def load_tokenizer(tokenizer_path,max_length):
    tokenizer = SimpleDNATokenizer(max_length)  # Update max_length if needed
    tokenizer.load_vocab(tokenizer_path)
    return tokenizer


def run(args, rank=None):
    assert args.batch_size % args.training_batch_size == 0
    set_seed(args.seed)
    args_dict = vars(args)
    exp_name = f'grad:{args.tweedie}-α:{args.alpha}-γ:{args.gamma}-M:{args.sample_M}-I:{args.inner_epochs}-B:{args.training_batch_size}-{args.batch_size}-S:{args.seed}-{args.tag}'
    wandb.init(
        project="DAV-DNA",
        job_type='FA',
        name=exp_name,
        config=args_dict
    )
    # os.environ["WANDB_MODE"] = "dryrun"

    if args.load_checkpoint_path:
        load_checkpoint_path = args.load_checkpoint_path
    else:
        load_checkpoint_path = None


    print("loading model")
    multi_model = False
    if args.model == 'enformer':
        # common_trunk = EnformerTrunk(n_conv=args.n_conv, channels=args.channels, n_transformers=args.n_transformers,
        #                              n_heads=args.n_heads, key_len=args.key_len,
        #                              attn_dropout=args.attn_dropout, pos_dropout=args.pos_dropout,
        #                              ff_dropout=args.ff_dropout, crop_len=args.crop_len)
        common_trunk = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                     attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        model = BaseModel(embedding=common_trunk, head=reg_head, cdq=args.cdq, batch_size=args.batch_size,
                          val_batch_num=1, task=args.task, n_tasks=args.n_task, saluki_body=args.saluki_body)
    elif args.model == 'multienformer':
        common_trunk = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                     attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        model = BaseModelMultiSep(embedding=common_trunk, head=reg_head, cdq=args.cdq, batch_size=args.batch_size, val_batch_num=args.val_batch_num)
        multi_model = True
    elif args.model == 'timedenformer':
        common_trunk = TimedEnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                     attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        model = BaseModel(embedding=common_trunk, head=reg_head, cdq=args.cdq, batch_size=args.batch_size,
                          val_batch_num=args.val_batch_num, timed=True)
    else:
        raise NotImplementedError

    if args.pre_model_path is not None:
        print("loading pretrained model: ", args.pre_model_path)
        model_path = args.pre_model_path
        model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'], strict=True)
    if load_checkpoint_path is not None:
        print("loading stored model: ", load_checkpoint_path)
        checkpoint = torch.load(load_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print('total params:', sum(p.numel() for p in model.parameters()))

    model.cuda()
    model.eval()

    timesteps = torch.linspace(1, 1e-5, model.ref_model.config.sampling.steps + 1)
    dt = (1 - 1e-5) / model.ref_model.config.sampling.steps
    
    for param in model.ref_model.parameters():
        param.requires_grad = True 

    policy_optimizer = torch.optim.AdamW(
        model.ref_model.parameters(), 
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    pretrained_model = copy.deepcopy(model.ref_model)
    evaluator = DNAQualityAssessor()
    model.set_double_reward_model() 

    for epoch in tqdm(range(args.epochs), desc="Training epochs: ", position=0):
        model.ref_model.eval()
        model.set_batch_size(args.batch_size)
        gen_samples, zero_shot_gen_samples, value_func_preds, reward_model_preds, eval_reward_model_preds, selected_baseline_preds, baseline_preds, eval_base_reward_model_preds, q_xs_history, x_history, q_x0_history = model.controlled_decode_rl(
            gen_batch_num=args.val_batch_num, 
            sample_M=args.sample_M, 
            options = args.tweedie,
            alpha = args.alpha,
            gamma = args.gamma
        )
     
        hepg2_values_ours = reward_model_preds.cpu().numpy()
        hepg2_values_baseline = baseline_preds.cpu().numpy()
        eval_hepg2_values_ours = eval_reward_model_preds.cpu().numpy()
        eval_hepg2_values_baseline = eval_base_reward_model_preds.cpu().numpy()

        division = 32
        samples_per_division = len(gen_samples) // division

        searched_div, searched_atac, searched_mer_corr = evaluator.evaluate(gen_samples)
        base_div, base_atac, base_mer_corr = evaluator.evaluate(zero_shot_gen_samples)
        model.ref_model.train()
    
        q_xs_history = torch.stack(q_xs_history)  
        q_x0_history = torch.stack(q_x0_history) 
        x_history = torch.stack(x_history)  

        print(f"==== train: hepg2_values_ours: {np.median(hepg2_values_ours)}")
        print(f"==== train: hepg2_values_baseline: {np.median(hepg2_values_baseline)}")
        print(f"==== eval : hepg2_values_ours: {np.median(eval_hepg2_values_ours)}")
        print(f"==== eval : hepg2_values_baseline: {np.median(eval_hepg2_values_baseline)}")

        for inner_epoch in range(args.inner_epochs):
            # 매 inner epoch마다 배치 차원에서 랜덤하게 섞기
            batch_size = q_xs_history.shape[1]
            shuffle_indices = torch.randperm(batch_size, device=q_xs_history.device)
            q_xs_shuffled = q_xs_history[:, shuffle_indices]
            q_x0_shuffled = q_x0_history[:, shuffle_indices]
            x_shuffled = x_history[:, shuffle_indices]
            
            for i in trange(division, desc=f"Policy Training Divisions (batch_size={len(gen_samples)//division})"):
                loss = 0.0  # float으로 초기화
                policy_optimizer.zero_grad()
                kl_loss = 0.0
                train_batch_size = len(gen_samples) // division
                q_xs_train = q_xs_shuffled[: ,i*train_batch_size:(i+1)*train_batch_size]
                q_x0_train = q_x0_shuffled[: ,i*train_batch_size:(i+1)*train_batch_size]
                x_train = x_shuffled[: ,i*train_batch_size:(i+1)*train_batch_size]

                for enum, (q_xs, q_x0, x, t) in tqdm(enumerate(zip(q_xs_train, q_x0_train, x_train, timesteps[:-1])), total=q_xs_train.shape[0], desc=f"Division {i+1}/{division} - Timesteps", leave=False):
                    # MLE loss
                    # min - (\pi_theta(x_{t-1}|x_t)))
                    sigma_t, _ = model.ref_model.noise(t * torch.ones(x.shape[0], device=x.device))
                    sigma_s, _ = model.ref_model.noise((t - dt) * torch.ones(x.shape[0], device=x.device))
                    move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
                    move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]
                    # negativity
                    weight = - (move_chance_t - move_chance_s) / (1 - move_chance_t)

                    copy_flag = (x != model.ref_model.mask_index).to(x.dtype)
                    logits = model.ref_model.backbone(x, sigma_t)
                    log_probs = model.ref_model._subs_parameterization(logits=logits, xt=x)
                    log_probs_selected = torch.gather(log_probs, -1, torch.argmax(q_x0, dim=-1, keepdim=True)).squeeze(-1)
                    
                    loss += copy_flag * weight * log_probs_selected.mean()

                    # KL divergence loss 
                    # min D_KL(\pi_theta(x_{t-1}|x_t) || \pi_pretrained(x_{t-1}|x_t))
                    with torch.no_grad():
                        logits_pretrained = pretrained_model.backbone(x, sigma_t)
                        log_probs_pretrained = pretrained_model._subs_parameterization(logits=logits_pretrained, xt=x)
                    log_p_theta = log_probs  # [batch, seq, vocab] (including mask token)
                    log_p_pretrained = log_probs_pretrained
                    
                    p_theta = log_p_theta.exp()
                    p_pretrained = log_p_pretrained.exp()
                    # 0 for unmasked positions, 1 for masked positions
                    mask_flag = (1 - copy_flag) 
                    mask_flag_expanded = mask_flag[:, :, None].expand_as(p_theta)
                    
                    # KL divergence: D_KL(p_theta || p_pretrained) = sum(p_theta * (log_p_theta - log_p_pretrained))
                    kl_per_token = p_theta * (log_p_theta - log_p_pretrained)
                    kl_masked = mask_flag_expanded * kl_per_token  # Only compute for masked positions
                    kl_loss += kl_masked.sum(dim=(1, 2)).mean()  # Sum over seq and vocab, then mean over batch

                loss = loss.mean()
                kl_loss = kl_loss.mean()
                total_loss = loss + kl_loss * args.alpha
                total_loss.backward()

                if (i + 1) * samples_per_division % args.training_batch_size == 0:
                    torch.nn.utils.clip_grad_norm_(model.ref_model.parameters(), max_norm=1.0)
                    policy_optimizer.step()
                    policy_optimizer.zero_grad()

        log_dict = {
            "reward/hepg2_values_searched": np.median(hepg2_values_ours), 
            "reward/hepg2_values_baseline": np.median(hepg2_values_baseline),
            "reward/eval_hepg2_values_searched": np.median(eval_hepg2_values_ours),
            "reward/eval_hepg2_values_baseline": np.median(eval_hepg2_values_baseline),
            "div/searched_div": searched_div,
            "atac/searched_atac": searched_atac,
            "mer_corr/searched_mer_corr": searched_mer_corr,
            "div/base_div": base_div,
            "atac/base_atac": base_atac,
            "mer_corr/base_mer_corr": base_mer_corr,
            "loss/total_loss": total_loss.item(),
            "loss/policy_loss": loss.item(),
            "loss/kl_loss": kl_loss.item(),
            "epoch": epoch
        }

        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args)
            }
            checkpoint_path = f"./checkpoints/{exp_name}/finetune_epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")


        if (epoch + 1) % 50 == 0:
            model.set_batch_size(args.eval_batch_size)
            gen_samples, zero_shot_gen_samples, value_func_preds, reward_model_preds, eval_reward_model_preds, selected_baseline_preds, baseline_preds, eval_base_reward_model_preds, q_xs_history, x_history, q_x0_history = model.controlled_decode_rl(
                gen_batch_num = args.val_batch_num, 
                sample_M=args.sample_M, 
                options = args.tweedie,
                alpha = args.alpha,
                gamma = args.gamma
            ) 
            del loss
            del q_xs_history
            del q_x0_history
            del x_history 
            gc.collect()

            searched_div, searched_atac, searched_mer_corr = evaluator.evaluate(gen_samples)
            base_div, base_atac, base_mer_corr = evaluator.evaluate(zero_shot_gen_samples)
            log_dict.update({
                "result/searched_div": searched_div,
                "result/searched_atac": searched_atac,
                "result/searched_mer_corr": searched_mer_corr,
                "result/base_div": base_div,
                "result/base_atac": base_atac,
                "result/base_mer_corr": base_mer_corr,
                "result/hepg2_values_searched": np.median(hepg2_values_ours),
                "result/hepg2_values_baseline": np.median(hepg2_values_baseline),
                "result/eval_hepg2_values_searched": np.median(eval_hepg2_values_ours),
                "result/eval_hepg2_values_baseline": np.median(eval_hepg2_values_baseline)
            })

        wandb.log(log_dict)


        


        np.savez( "./log/%s-%s_tw" %(args.task, args.reward_name), decoding = hepg2_values_ours, baseline = hepg2_values_baseline)


    wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    parser.add_argument('--task', type=str, default="rna_saluki",
                        help="task", required=False)
    parser.add_argument('--saluki_body', type=int, default=0,
                        required=False)
    parser.add_argument('--n_task', type=int, default=1,
                        help="number of task head", required=False)
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="name of the dataset to train on", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default=0, help="number of properties to use for condition",
                        required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--model', type=str, default='enformer',
                        help="name of the model", required=False)
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help="name of the tokenizer", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=1,
                        help="total epochs", required=False)
    parser.add_argument('--max_iters', type=int, default=50000,
                        help="total iterations", required=False)
    parser.add_argument('--batch_size', type=int, default=256,
                        help="batch size", required=False)
    parser.add_argument('--sample_M', type=int, default=20,
                        help="sample width", required=False)
    parser.add_argument('--val_batch_num', type=int, default=1,
                        help="val batches", required=False)
    parser.add_argument('--num_workers', type=int, default=12,
                        help="number of workers for data loaders", required=False)
    parser.add_argument('--save_start_epoch', type=int, default=120,
                        help="save model start epoch", required=False)
    parser.add_argument('--save_interval_epoch', type=int, default=10,
                        help="save model epoch interval", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)
    parser.add_argument('--max_len', type=int, default=512,
                        help="max_len", required=False)
    parser.add_argument('--seed', type=int, default=44,
                        help="seed", required=False)
    parser.add_argument('--reward_name', type=str, default='HepG2',
                        help="Plot Y axis name", required=False)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0,
                        help="gradient norm clipping. smaller values mean stronger normalization.", required=False)
    parser.add_argument('--auto_fp16to32', action='store_true',
                        default=False, help='Auto casting fp16 tensors to fp32 when necessary')
    parser.add_argument('--load_checkpoint_path', type=str, default=None,
                        help="Path to load training checkpoint (if resuming training)", required=False)
    parser.add_argument('--pre_root_path', default=None,
                        help="Path to the pretrain data directory", required=False)
    parser.add_argument('--pre_model_path', default=None,
                        help="Path to the pretrain model", required=False)
    parser.add_argument('--root_path', type=str, default='/home/lix361/projects/rna_optimization/generative/5UTR_Ensembl_cond',
                        help="Path to the root data directory", required=False)
    parser.add_argument('--output_tokenizer_dir', type=str,
                        default='/home/lix361/projects/rna_optimization/generative/storage/5UTR_Ensembl_cond_seq/tokenizer',
                        help="Path to the saved tokenizer directory", required=False)
    parser.add_argument('--fix_condition', default=None,
                        help="fixed condition num", required=False)
    parser.add_argument('--conditions_path', default=None,
                        help="Path to the generation condition", required=False)
    parser.add_argument('--conditions_split_id_path', default=None,
                        help="Path to the conditions_split_id", required=False)
    parser.add_argument('--cdq', action='store_true',
                        default=False, help='CD-Q')
    parser.add_argument('--dist', action='store_true',
                        default=False, help='use torch.distributed to train the model in parallel')

    parser.add_argument('--epochs', type=int, default=50,
                        help="number of epochs", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=1e-4, help="learning rate", required=False)
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="coefficient for the kl regularization", required=False)
    parser.add_argument('--gamma', type=float, default=1.0,
                        help="coefficient for the discount factor", required=False)
    parser.add_argument('--tweedie',type=str,  default = True, help='gradient guidance', required=True)
    parser.add_argument("--training_batch_size", type=int, default=64, help="batch division", required=False)
    parser.add_argument("--inner_epochs", type=int, default=1, help="inner epochs", required=False)
    parser.add_argument("--tag", type=str, default="", help="tag", required=False)
    parser.add_argument("--eval_batch_size", type=int, default=640, help="eval batch size", required=False)
    parser.add_argument('--use_kl', action='store_true',
                        default=False, help='use kl divergence loss')

    args = parser.parse_args()

    run(args)
