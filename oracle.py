import wandb
import torch
import grelu
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path

from grelu.lightning import LightningModel
# from grelu.resources import artifacts, get_model_by_dataset, get_dataset_by_model
import grelu.data.preprocess
# from grelu.lightning import LightningModel
import grelu.data.dataset
import dataloader_gosai
import numpy as np
from typing import Callable, Union, List
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
import os
from polyleven import levenshtein
import itertools


# Gosai dataset

DEFAULT_WANDB_ENTITY = 'wangc239'
DEFAULT_WANDB_HOST = 'https://genentech.wandb.io'

def _check_wandb(host=DEFAULT_WANDB_HOST):
    pass
    # assert wandb.login(host=host), f'Weights & Biases (wandb) is not configured, see {DEFAULT_WANDB_HOST}/authorize'

def get_artifact(name, project, alias='latest'):
    _check_wandb()
    project_path = f'{DEFAULT_WANDB_ENTITY}/{project}'
    
    api = wandb.Api()    
    return api.artifact(f'{project_path}/{name}:{alias}')

def get_model_by_dataset_personal(dataset_name, project, alias='latest'):
    art = get_artifact(dataset_name, project, alias=alias)
    runs = art.used_by()
    assert len(runs) > 0
    return [x.name for x in runs[0].logged_artifacts()]

def get_dataset_by_model_personal(model_name, project, alias='latest'):
    art = get_artifact(model_name, project, alias=alias)
    run = art.logged_by()
    return [x.name for x in run.used_artifacts()]

def load_model_personal(project, model_name, alias='latest', checkpoint_file='model.ckpt', temp_dir='/data/wangc239/tmp'):

    art = get_artifact(model_name, project, alias=alias)

    with TemporaryDirectory(dir=temp_dir) as d:
        art.download(d)
        model = LightningModel.load_from_checkpoint(Path(d) / checkpoint_file, map_location='cuda')

    return model

def get_gosai_oracle(from_local=True):
    if from_local:
        model_load = LightningModel.load_from_checkpoint("/data/masatoshi/model.ckpt", map_location='cuda')
    else:
        pass
        #model_load = load_model_personal('human-mpra-gosai-2023', 'model')
    return model_load

def cal_gosai_pred(seqs, model=None):
    """
    seqs: list of sequences (detokenized ACGT...)
    """
    if model is None:
        model = get_gosai_oracle()
    df_seqs = pd.DataFrame(seqs, columns=['seq'])
    pred_dataset = grelu.data.dataset.DFSeqDataset(df_seqs)
    preds = model.predict_on_dataset(pred_dataset, devices=[0])
    return preds.squeeze() # numpy array with shape [n_seqs, 3]


def count_kmers(seqs, k=3):
    counts = {}
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            subseq = seq[i : i + k]
            try:
                counts[subseq] += 1
            except KeyError:
                counts[subseq] = 1
    return counts


def subset_for_eval(n=5000, seed=0):
    train_set, valid_set, test_set = dataloader_gosai.get_datasets_gosai()
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_set_sp = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), n, replace=False))
    valid_set_sp = torch.utils.data.Subset(valid_set, np.random.choice(len(valid_set), n, replace=False))
    test_set_sp = torch.utils.data.Subset(test_set, np.random.choice(len(test_set), n, replace=False))
    return train_set_sp, valid_set_sp, test_set_sp


def subset_eval_groundtruth(sets_sp):
    train_set_sp, valid_set_sp, test_set_sp = sets_sp
    train_set_sp_clss = train_set_sp.dataset.clss[train_set_sp.indices]
    valid_set_sp_clss = valid_set_sp.dataset.clss[valid_set_sp.indices]
    test_set_sp_clss = test_set_sp.dataset.clss[test_set_sp.indices]
    return train_set_sp_clss, valid_set_sp_clss, test_set_sp_clss


def subset_eval_preds(sets_sp, oracle_model=None):
    train_set_sp, valid_set_sp, test_set_sp = sets_sp
    train_preds = cal_gosai_pred(
        dataloader_gosai.batch_dna_detokenize(train_set_sp.dataset.seqs[train_set_sp.indices].numpy()), oracle_model)
    valid_preds = cal_gosai_pred(
        dataloader_gosai.batch_dna_detokenize(valid_set_sp.dataset.seqs[valid_set_sp.indices].numpy()), oracle_model)
    test_preds = cal_gosai_pred(
        dataloader_gosai.batch_dna_detokenize(test_set_sp.dataset.seqs[test_set_sp.indices].numpy()), oracle_model)
    return train_preds, valid_preds, test_preds


def subset_eval_kmers(sets_sp, k=3):
    train_set_sp, valid_set_sp, test_set_sp = sets_sp
    train_seqs = dataloader_gosai.batch_dna_detokenize(train_set_sp.dataset.seqs[train_set_sp.indices].numpy())
    valid_seqs = dataloader_gosai.batch_dna_detokenize(valid_set_sp.dataset.seqs[valid_set_sp.indices].numpy())
    test_seqs = dataloader_gosai.batch_dna_detokenize(test_set_sp.dataset.seqs[test_set_sp.indices].numpy())
    train_kmers = count_kmers(train_seqs, k)
    valid_kmers = count_kmers(valid_seqs, k)
    test_kmers = count_kmers(test_seqs, k)
    return train_kmers, valid_kmers, test_kmers


def subset_eval_embs(sets_sp, oracle_model=None):
    train_set_sp, valid_set_sp, test_set_sp = sets_sp
    train_sp_emb = cal_gosai_emb(
        dataloader_gosai.batch_dna_detokenize(train_set_sp.dataset.seqs[train_set_sp.indices].numpy()), oracle_model)
    valid_sp_emb = cal_gosai_emb(
        dataloader_gosai.batch_dna_detokenize(valid_set_sp.dataset.seqs[valid_set_sp.indices].numpy()), oracle_model)
    test_sp_emb = cal_gosai_emb(
        dataloader_gosai.batch_dna_detokenize(test_set_sp.dataset.seqs[test_set_sp.indices].numpy()), oracle_model)
    return train_sp_emb, valid_sp_emb, test_sp_emb


def cal_emb_pca(valid_set, n_components=50, oracle_model=None):
    # use valid_set not train_set because train_set is too large
    valid_all_emb = cal_gosai_emb(
        dataloader_gosai.batch_dna_detokenize(valid_set.seqs.numpy()), oracle_model)
    # pca on valid_all_emb
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(valid_all_emb.reshape(valid_all_emb.shape[0], -1))
    return pca


def subset_eval_embs_pca(sets_sp, pca, oracle_model=None):
    train_sp_emb, valid_sp_emb, test_sp_emb = subset_eval_embs(sets_sp, oracle_model)
    train_sp_emb_pca = pca.transform(train_sp_emb.reshape(train_sp_emb.shape[0], -1))
    valid_sp_emb_pca = pca.transform(valid_sp_emb.reshape(valid_sp_emb.shape[0], -1))
    test_sp_emb_pca = pca.transform(test_sp_emb.reshape(test_sp_emb.shape[0], -1))
    return train_sp_emb_pca, valid_sp_emb_pca, test_sp_emb_pca


# https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/flow_utils.py
def get_wasserstein_dist(embeds1, embeds2):
    if np.isnan(embeds2).any() or np.isnan(embeds1).any() or len(embeds1) == 0 or len(embeds2) == 0:
        return float('nan')
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return dist


def embed_on_dataset(
    model,
    dataset: Callable,
    devices: Union[str, int, List[int]] = "cpu",
    num_workers: int = 1,
    batch_size: int = 256,
):
    """
    Return embeddings for a dataset of sequences

    Args:
        dataset: Dataset object that yields one-hot encoded sequences
        devices: Device IDs to use
        num_workers: Number of workers for data loader
        batch_size: Batch size for data loader

    Returns:
        Numpy array of shape (B, T, L) containing embeddings.
    """
    torch.set_float32_matmul_precision("medium")

    # Make dataloader
    dataloader = model.make_predict_loader(
        dataset, num_workers=num_workers, batch_size=batch_size
    )

    # Get device
    orig_device = model.device
    device = model.parse_devices(devices)[1]
    if isinstance(device, list):
        device = device[0]
    #     warnings.warn(
    #         f"embed_on_dataset currently only uses a single GPU: {device}"
    #     )
    model.to(device)

    # Get embeddings
    preds = []
    model.model = model.model.eval()
    for batch in iter(dataloader):
        batch = batch.to(device)
        preds.append(model.model.embedding(batch).detach().cpu())

    # Return to original device
    model.to(orig_device)
    return torch.vstack(preds).numpy()


def cal_gosai_emb(seqs, model=None):
    """
    seqs: list of sequences (detokenized ACGT...)
    """
    if model is None:
        model = get_gosai_oracle()
    df_seqs = pd.DataFrame(seqs, columns=['seq'])
    pred_dataset = grelu.data.dataset.DFSeqDataset(df_seqs)
    embs = embed_on_dataset(model, pred_dataset, devices=[0])
    return embs # numpy array with shape [n_seqs, 3072, 2]


def cal_atac_pred_new(seqs, model=None):
    """
    seqs: list of sequences (detokenized ACGT...)
    """
    if model is None:
        model = LightningModel.load_from_checkpoint(os.path.join(base_path, 'mdlm/gosai_data/binary_atac_cell_lines.ckpt'), map_location='cuda')
    model.eval()
    tokens = dataloader_gosai.batch_dna_tokenize(seqs)
    tokens = torch.tensor(tokens).long().cuda()
    onehot_tokens = torch.nn.functional.one_hot(tokens, num_classes=4).float()
    preds = model(onehot_tokens.float().transpose(1, 2)).detach().cpu().numpy()
    return preds.squeeze() # numpy array with shape [n_seqs, 7]

def compare_kmer(kmer1, kmer2, n_sp1, n_sp2):
    kmer_set = set(kmer1.keys()) | set(kmer2.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, kmer in enumerate(kmer_set):
        if kmer in kmer1:
            counts[i][1] = kmer1[kmer] * n_sp2 / n_sp1
        if kmer in kmer2:
            counts[i][0] = kmer2[kmer]

    return pearsonr(counts[:, 0], counts[:, 1])

def edit_dist(seq1, seq2):
    return levenshtein(seq1, seq2) / 1

def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)   


def levenshtein_diversity(seqs):
    if isinstance(seqs, list):
        seqs = torch.stack(seqs)
    seqs = dataloader_gosai.batch_dna_detokenize(seqs.cpu())
    return mean_pairwise_distances(seqs)


def cal_highexp_kmers(k=3, return_clss=False):
    train_set = dataloader_gosai.get_datasets_gosai(skip_valid=True)[0]
    exp_threshold = np.quantile(train_set.clss[:, 0].numpy(), 0.99) # 4.56
    highexp_indices = [i for i, data in enumerate(train_set) if data['clss'][0] > exp_threshold]
    highexp_set_sp = torch.utils.data.Subset(train_set, highexp_indices)
    highexp_seqs = dataloader_gosai.batch_dna_detokenize(highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy())
    highexp_kmers_99 = count_kmers(highexp_seqs, k=k)
    n_highexp_kmers_99 = len(highexp_indices)

    exp_threshold = np.quantile(train_set.clss[:, 0].numpy(), 0.999) # 6.27
    highexp_indices = [i for i, data in enumerate(train_set) if data['clss'][0] > exp_threshold]
    highexp_set_sp = torch.utils.data.Subset(train_set, highexp_indices)
    highexp_seqs = dataloader_gosai.batch_dna_detokenize(highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy())
    highexp_kmers_999 = count_kmers(highexp_seqs, k=k)
    n_highexp_kmers_999 = len(highexp_indices)
        
    return highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999


def cal_atac_pred_new(seqs, model=None):
    """
    seqs: list of sequences (detokenized ACGT...) or tokenized tensor
    """
    if model is None:
        model = LightningModel.load_from_checkpoint(os.path.join('/home/son9ih/dav-discrete/artifacts/ATAC_oracle/binary_atac_cell_lines.ckpt'), map_location='cuda')
    # model.cuda()/
    model.eval()
    
    if isinstance(seqs, list):
        tokens = torch.stack(seqs)
    else:
        tokens = seqs

    tokens = tokens.long().cuda()
    onehot_tokens = torch.nn.functional.one_hot(tokens, num_classes=4).float()
    # error
    preds = model(onehot_tokens.float().transpose(1, 2)).detach().cpu().numpy()
    return (preds > 0.5).mean().item()


def compare_kmer(kmer1, kmer2, n_sp1, n_sp2):
    kmer_set = set(kmer1.keys()) | set(kmer2.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, kmer in enumerate(kmer_set):
        if kmer in kmer1:
            counts[i][1] = kmer1[kmer] * n_sp2 / n_sp1
        if kmer in kmer2:
            counts[i][0] = kmer2[kmer]
    return pearsonr(counts[:, 0], counts[:, 1])[0]


class DNAQualityAssessor:
    def __init__(self):
        self.diversity = levenshtein_diversity
        self.atac_acc = cal_atac_pred_new
        self.mer_corr = compare_kmer

        self.atac_model = LightningModel.load_from_checkpoint(os.path.join('/home/son9ih/dav-discrete/artifacts/ATAC_oracle/binary_atac_cell_lines.ckpt'), map_location='cuda')
        _, _, self.highexp_kmers_999, self.n_highexp_kmers_999 = cal_highexp_kmers(k=3, return_clss=False)

    def evaluate(self, seqs):
        diversity = self.diversity(seqs)
        atac = self.atac_acc(seqs, self.atac_model)
        if isinstance(seqs, list):
            seqs = torch.stack(seqs)
        seqs = dataloader_gosai.batch_dna_detokenize(seqs.cpu())
        kmers = count_kmers(seqs, k=3)
        mer_corr = self.mer_corr(self.highexp_kmers_999, kmers, self.n_highexp_kmers_999, len(seqs))
        return diversity, atac, mer_corr