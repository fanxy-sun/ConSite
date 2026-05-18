import os, inspect
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Sequence, Literal, Any
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_scatter import scatter_mean
import torch_geometric

import esm
from esm.constants import proteinseq_toks
import dgl
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from rdkit import Chem
from torchdrug import data, utils, transforms
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa, three_to_one
from Bio import pairwise2
from Bio import BiopythonDeprecationWarning

import itertools
import logging
import warnings
import time
from copy import deepcopy
from functools import partial
import hashlib
from tqdm import tqdm as top_tqdm
from collections.abc import Mapping
from pandarallel import pandarallel
from concurrent.futures import ProcessPoolExecutor, as_completed
import string
import pickle
from collections import defaultdict, deque
import json
import ast

from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from rxnfp.transformer_fingerprints import (RXNBERTFingerprintGenerator, get_default_model_and_tokenizer,)

from common.utils import build_data_pools, process_row



RawMSA = Sequence[Tuple[str, str]]
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


# =========================================== Top@RFDiffsuion ======================================================
num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    'UNK','MAS',
    ]
aa2num= {x:i for i,x in enumerate(num2aa)}

# full sc atom representation (Nx14)
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1"," HE2",  None,  None,  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HD2"," HE1"," HE2"," HZ ",  None,  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HE2"," HD2"," HH ",  None,  None,  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # unk
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # mask
]
# =========================================== Bottom@RFDiffsuion ===================================================

# =========================================== Top@RosettaFold ====================================================== 
PARAMS = {
    "DMIN"    : 2.0,
    "DMAX"    : 20.0,
    "DBINS"   : 36,
    "ABINS"   : 36,
}


def get_pair_dist(a, b):
    """calculate pair distances between two sets of points
    
    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist


def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c 

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v /= torch.norm(v, dim=-1, keepdim=True)
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v*w, dim=-1)

    return torch.acos(vw)


def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)

    return torch.atan2(y, x)

def xyz_to_c6d(xyz, params=PARAMS):
    """convert cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz : pytorch tensor of shape [batch,nres,3,3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    Returns
    -------
    c6d : pytorch tensor of shape [batch,nres,nres,4]
          stores stacked dist,omega,theta,phi 2D maps 
    """
    
    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[:,:,0]
    Ca = xyz[:,:,1]
    C  = xyz[:,:,2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca    

    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = torch.zeros([batch,nres,nres,4],dtype=xyz.dtype,device=xyz.device)

    dist = get_pair_dist(Cb,Cb)
    dist[torch.isnan(dist)] = 999.9
    c6d[...,0] = dist + 999.9*torch.eye(nres,device=xyz.device)[None,...]
    b,i,j = torch.where(c6d[...,0]<params['DMAX'])

    c6d[b,i,j,torch.full_like(b,1)] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
    c6d[b,i,j,torch.full_like(b,2)] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    c6d[b,i,j,torch.full_like(b,3)] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

    # fix long-range distances
    c6d[...,0][c6d[...,0]>=params['DMAX']] = 999.9
    
    mask = torch.zeros((batch, nres,nres), dtype=xyz.dtype, device=xyz.device)
    mask[b,i,j] = 1.0
    return c6d, mask
    
def xyz_to_t2d(xyz_t, t0d, params=PARAMS):
    """convert template cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz_t : pytorch tensor of shape [batch,templ,nres,3,3]
            stores Cartesian coordinates of template backbone N,Ca,C atoms
    t0d:  0-D template features (HHprob, seqID, similarity) [batch, templ, 3]

    Returns
    -------
    t2d : pytorch tensor of shape [batch,nres,nres,1+6+3]
          stores stacked dist,omega,theta,phi 2D maps 
    """
    B, T, L = xyz_t.shape[:3]
    c6d, mask = xyz_to_c6d(xyz_t.view(B*T,L,3,3), params=params)
    c6d = c6d.view(B, T, L, L, 4)
    mask = mask.view(B, T, L, L, 1)
    #
    dist = c6d[...,:1]*mask / params['DMAX'] # from 0 to 1 # (B, T, L, L, 1)
    dist = torch.clamp(dist, 0.0, 1.0)
    orien = torch.cat((torch.sin(c6d[...,1:]), torch.cos(c6d[...,1:])), dim=-1)*mask # (B, T, L, L, 6)
    t0d = t0d.unsqueeze(2).unsqueeze(3).expand(-1, -1, L, L, -1)
    #
    t2d = torch.cat((dist, orien, t0d), dim=-1)
    t2d[torch.isnan(t2d)] = 0.0
    return t2d


# =========================================== Bottom@RosettaFold =================================================== 

class Alphabet(object):
    """
    增强版 class Alphabet
    """
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, truncation_seq_length: int = None):
        if self.use_msa:
            return MSABatchConverter(self, truncation_seq_length)
        else:
            return BatchConverter(self, truncation_seq_length)

    def get_sample_converter(self, truncation_seq_length: int = None):
        """
        针对一个样本：
            * 按truncation_seq_length截断过长序列
            * 氨基酸字母到数字映射
            * 不添加cls/bos, eos, pad，将这些操作留到collate函数
        """
        return SampleConverter(self, truncation_seq_length)
    
    def get_sample_msa_converter(self, truncation_seq_length: int = None):
        """
        针对一个样本对应的MSA：
            * 按truncation_seq_length截断MSA中每个过长序列
            * 氨基酸字母到数字映射
            * 不添加cls/bos, eos, pad，将这些操作留到collate函数
        """
        return SampleMSAConverter(self, truncation_seq_length)
    

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("ESM-1", "protein_bert_base"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif name in ("MSA Transformer", "msa_transformer"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True           # 这里原来是False
            use_msa = True
        elif "invariant_gvp" in name.lower():
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<cath>", "<af2>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]



class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [seq_str[:self.truncation_seq_length] for seq_str in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens



class MSABatchConverter(BatchConverter):
    def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens = super().__call__(msa)
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens

        return labels, strs, tokens



class SampleConverter:
    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, seq_str: str) -> Tuple[str, torch.Tensor]:
        seq_encoded = self.alphabet.encode(seq_str)
        if self.truncation_seq_length:
            seq_encoded = seq_encoded[:self.truncation_seq_length]
        seq_tensor = torch.tensor(seq_encoded, dtype=torch.int64)
        return seq_str, seq_tensor
    


class SampleMSAConverter:
    """
    Converts multiple MSA sequences to tokenized tensors without special tokens or padding.
    
    This class handles tokenization and truncation for all sequences in an MSA,
    without adding special tokens like BOS/EOS or performing padding.
    These operations are left for the collate function.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        """
        Initialize the converter.
        
        Args:
            alphabet: The alphabet for amino acid to index conversion
            truncation_seq_length: Maximum allowed sequence length
        """
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, sequences: List[str]) -> List[torch.Tensor]:
        """
        Convert a list of MSA sequences to token tensors without padding.
        
        Args:
            sequences: List of protein sequence strings from an MSA
            
        Returns:
            List of tensors, where each tensor represents a tokenized sequence
            from the MSA without any padding or special tokens
        """
        if not sequences:
            return []
        
        # Encode and truncate each sequence
        tokenized_seqs = []
        for seq in sequences:
            # Tokenize sequence
            seq_encoded = self.alphabet.encode(seq)
            
            # Apply truncation if needed
            if self.truncation_seq_length:
                seq_encoded = seq_encoded[:self.truncation_seq_length]
                
            # Convert to tensor and add to list
            seq_tensor = torch.tensor(seq_encoded, dtype=torch.int64)
            tokenized_seqs.append(seq_tensor)
        
        return tokenized_seqs
    


class MatrixLinearNormalizer(nn.Module):
    """
    更多信息：https://chatgpt.com/c/686bfd7d-0154-8005-9f7f-3d25659c6e66


    Learnable (affine) normalizer for pair‑wise matrices with optional symmetrisation & APC pre‑processing and several *classic* normalisation utilities.

    This class provides methods to standardize and compare different types of structural matrices:
    - Attention maps (from transformer models like ESM)
    - Contact maps (predicted protein residue contacts)
    - Evolutionary coupling matrices (from DCA methods like CCMpred)

    一般来说，规范化处理有下面几种：
        * 对称化（Symmetrize）
        * APC 校正（Average Product Correction）
        * 归一化（Normalization）
        * 按max_seq_length截断（这一操作不在本类中实现）
        * 去除special tokens（这一操作不在本类中实现）

    Attention maps, Contact maps, Evolutionary coupling matrices的映射层是否共用：
        * Attention → 单独一套参数
        * Concat → 单独一套参数
        * Coupling 
            * Bias-injection → 单独一套参数
            * Aux-loss → 单独一套参数

    映射目标区间：
        * Attention / Concat → [0,1]
        * Coupling：两份输出 [0,1] & [-1,1]，[0,1] 便于与注意力作 KL / MSE 对齐，[-1,1] 专供 bias-injection，保持正负号表示协同 / 排斥。社区（Direct coupling analysis and the attention mechanism）常用做法是 tanh 将原始耦合拉到 [-1,1] 后直接加到 QKᵀ 上。
    
    关于激活后是否破坏语义：
        * Bias-injection 分支：不用非线性；
        * Aux-loss 分支：可用 sigmoid / tanh
        偏置注入需要保留符号与相对幅度，非线性会压缩区分度；而在辅助损失路径，轻微压缩可减小极端梯度，Sigmoid→概率、Tanh→有符号相关性。

    关于如何选择激活函数：
        * Attention → sigmoid
        * Concat → sigmoid
        * Coupling 
            * Bias-injection → identity
            * Aux-loss → tanh

    """

    def __init__(
        self,
        activation: Literal['sigmoid', 'tanh', 'identity'] = 'sigmoid',
        apply_symmetrize: bool = True,
        apply_apc: bool = True,
        init_scale: float = 1.0,
        init_bias: float = 0.0,
        target_range: Tuple[float, float] = (0.0, 1.0),
        clamp_weight: Optional[Tuple[float, float]] = (0.1, 10.0),
    ) -> None:
        '''
        Args:
            activation : {'sigmoid', 'tanh', 'identity'}, default 'sigmoid'. Activation applied after the affine transform.  
            apply_symmetrize : bool, default True. Whether to make each input matrix symmetric *before* the affine transform.
            apply_apc : bool, default True. Whether to apply Average Product Correction (APC) *before* the affine transform.
            init_scale / init_bias : float, default (1.0, 0.0). Initial parameters for the affine transform y = w * x + b.
            target_range : 2‑tuple of float, default (0.0, 1.0). Clamp the *final* output into this range (after activation).
            clamp_weight : 2‑tuple of float or None, default (0.1, 10.0). If not None, ``weight`` is clipped to this interval *each* forward
                pass when ``self.training`` is True – a cheap but effective way to keep gradients in check.
        '''
        super().__init__()

        # Learnable affine parameters (scalar -> broadcast to whole matrix)
        self.weight = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias: nn.Parameter = nn.Parameter(torch.tensor(float(init_bias)))

        if activation not in {'sigmoid', 'tanh', 'identity'}:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activation

        self.apply_symmetrize = apply_symmetrize
        self.apply_apc = apply_apc
        self.target_range = target_range
        self._clamp_weight = clamp_weight

        # Register penalty as a buffer to persist across checkpoints
        self.register_buffer('_penalty', torch.tensor(0.0))



    def forward(self, matrix: torch.Tensor, collect_penalty: bool = True) -> torch.Tensor:  
        """Normalise **one or a batch** (B×L×L or L×L) matrices."""
        orig_dim = matrix.dim()
        if orig_dim == 2:
            matrix = matrix.unsqueeze(0)  # → [1, L, L]

        # Optional fixed pre‑processing
        if self.apply_symmetrize:
            matrix = self.symmetrize_matrix(matrix)
        if self.apply_apc:
            matrix = self.apply_apc_correction(matrix)

        # Parameter clamping for stability
        if self.training and self._clamp_weight is not None:
            with torch.no_grad():
                self.weight.data = self.weight.data.clamp(self._clamp_weight[0], self._clamp_weight[1])

        # Learnable affine transform
        # Ensure weight and bias are on the same device as input matrix
        weight = self.weight.to(matrix.device)
        bias = self.bias.to(matrix.device)
        out = weight * matrix + bias

        # Non‑linearity
        if self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.activation == 'tanh':
            out = torch.tanh(out)
        # else 'identity' – do nothing

        # 仅记录，不裁剪
        if collect_penalty and self.target_range != (-float('inf'), float('inf')):
            low, high = self.target_range
            overflow  = (out - high).clamp_min(0)
            underflow = (low - out).clamp_min(0)
            self._penalty.copy_((overflow.pow(2) + underflow.pow(2)).mean())
        else:
            self._penalty.copy_(torch.tensor(0., device=out.device))

        # Remove fake batch dim if needed
        if orig_dim == 2:
            out = out.squeeze(0)
        return out


    def range_penalty(self) -> torch.Tensor:
        return getattr(self, '_penalty', torch.tensor(0., device=self.weight.device))


    @staticmethod
    def standardize_attention_maps(
        attention_maps: torch.Tensor,
        normalize: bool = False,
        symmetrize: bool = True,
        apply_apc: bool = True,
        normalization_method: str = 'minmax',
        target_range: Tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """
        Standardize attention maps for contact prediction or comparison.
        
        Attention maps from ESM have these characteristics:
        - Values generally in [0,1] range after softmax
        - Sum to 1 along source dimension
        - May not be symmetric by default
        
        Parameters:
            attention_maps: 
                * 处理concat map时：批次内一个样本的attention maps，形状为[layers*heads, valid_len, valid_len]
                * 处理attention map时：2D张量 [valid_len, valid_len]
                * 处理coupling map时：2D张量 [valid_len, valid_len]，不过在传入standardize_attention_maps前进行unsquence伪装
            normalize: Whether to apply normalization
            symmetrize: Whether to make maps symmetric (recommended for structure prediction)
            apply_apc: Whether to apply Average Product Correction (recommended)
            normalization_method: Method for normalization if normalize=True
            target_range: Target range for minmax normalization
            
        Returns:
            Standardized attention maps：本函数不改变输入张量的形状
        """
        is_2d_input = (attention_maps.dim() == 2)

        # For 2D input, unsqueeze to add a batch dimension temporarily
        if is_2d_input:
            result = attention_maps.clone().unsqueeze(0)  # [1, valid_len, valid_len]
        else:
            result = attention_maps.clone()  # [layers*heads, valid_len, valid_len] or [batch_size, valid_len, valid_len]


        if symmetrize:
            # For 2D input, symmetrize directly without batch dimension
            if is_2d_input:
                result = MatrixLinearNormalizer.symmetrize_matrix(result.squeeze(0)).unsqueeze(0)
            else:
                result = MatrixLinearNormalizer.symmetrize_matrix(result)
        
        # Apply APC correction if requested
        # Treat first dimension (layers*heads) as batch dimension for APC function
        if apply_apc:
            result = MatrixLinearNormalizer.apply_apc_correction(result)
        
        # Apply normalization if requested
        # Treat first dimension (layers*heads) as batch dimension for normalization functions
        if normalize:
            if normalization_method == "minmax":
                result = MatrixLinearNormalizer.min_max_normalize(result, target_range)
            elif normalization_method == "zscore":
                result = MatrixLinearNormalizer.zscore_normalize(result)
            elif normalization_method == "rank":
                result = MatrixLinearNormalizer.rank_normalize(result)

        # For 2D input, remove the temporary batch dimension
        if is_2d_input:
            result = result.squeeze(0)

        return result

    @staticmethod
    def min_max_normalize(tensor: torch.Tensor, target_range: Tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
        """
        Min-max normalize a tensor to the target range.

        必须是至少2维的张量，第一维是batch维度，后续维度任意。每个batch样本独立计算min/max并归一化，零值被视为padding跳过处理。
        
        Good for:
        - Normalizing coupling matrices with arbitrary scales
        - Standardizing matrices for visualization
        - Preparing matrices for comparison when absolute values differ
        
        Less necessary for:
        - Contact maps (already in [0,1])
        - Attention maps (already normalized by softmax)
        """
        batch_size = tensor.shape[0]
        result = torch.clone(tensor)
        
        for b in range(batch_size):
            # Calculate min/max, excluding padding (zero values)
            mask = tensor[b] != 0
            if not mask.any():
                continue
                
            t_min = tensor[b][mask].min()
            t_max = tensor[b][mask].max()
            
            # Skip normalization if min equals max
            if t_min == t_max:
                continue
                
            # Apply min-max normalization
            result[b][mask] = (tensor[b][mask] - t_min) / (t_max - t_min) * (target_range[1] - target_range[0]) + target_range[0]
        
        return result
    

    @staticmethod
    def zscore_normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Z-score normalize a tensor (mean=0, std=1).
        
        Good for:
        - Statistical analysis
        - Comparing distributions across different proteins
        - Highlighting outliers
        
        Less useful for:
        - Direct contact prediction
        - Visualizing contact patterns
        """
        batch_size = tensor.shape[0]
        result = torch.clone(tensor)
        
        for b in range(batch_size):
            # Calculate mean/std, excluding padding (zero values)
            mask = tensor[b] != 0
            if not mask.any():
                continue
                
            mean = tensor[b][mask].mean()
            std = tensor[b][mask].std() + eps
            
            # Apply z-score normalization
            result[b][mask] = (tensor[b][mask] - mean) / std
        
        return result
    

    @staticmethod
    def rank_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor using percentile ranks.
        
        Good for:
        - Comparing matrices with very different distributions
        - Reducing the influence of outliers
        - When absolute values are not meaningful
        
        Less useful for:
        - When absolute values carry important information
        - When the distribution shape matters
        """
        batch_size = tensor.shape[0]
        result = torch.clone(tensor)
        
        for b in range(batch_size):
            # Get non-zero elements (excluding padding)
            mask = tensor[b] != 0
            if not mask.any():
                continue
                
            # Get values and compute ranks
            values = tensor[b][mask].flatten()
            ranks = values.argsort().argsort().float() / (values.numel() - 1)
            
            # Replace values with normalized ranks
            result[b][mask] = ranks.reshape(-1)[torch.arange(mask.sum())]
        
        return result
    

    @staticmethod
    def apply_apc_correction(tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply Average Product Correction to reduce indirect correlations.
        
        The APC method removes background correlation by subtracting the expected
        correlation due to shared interactions with other positions.
        
        This implementation uses the standard APC formula without masking,
        which is appropriate for attention maps and coupling matrices where
        all values are meaningful.
        
        Good for:
        - DCA coupling matrices (essential to remove indirect correlations)
        - Attention maps (improves contact prediction)
        - Any correlation matrix where indirect effects exist
        
        Less necessary for:
        - Contact maps that already incorporate APC
        - Matrices where indirect correlations are desired
        
        Args:
            tensor: Input tensor of shape [batch_size, height, width] or [height, width]
                   For batch processing, applies APC to each matrix in the batch
        
        Returns:
            APC-corrected tensor with same shape as input
        """
        # Handle both batched and single matrix inputs
        if tensor.dim() == 2:
            # Single matrix case
            a1 = tensor.sum(-1, keepdims=True)  # Row sums
            a2 = tensor.sum(-2, keepdims=True)  # Column sums
            a12 = tensor.sum((-1, -2), keepdims=True)  # Total sum
            
            avg = a1 * a2
            avg.div_(a12)  # in-place to reduce memory
            normalized = tensor - avg
            return normalized
        
        elif tensor.dim() == 3:
            # Batch processing case
            batch_size = tensor.shape[0]
            result = torch.zeros_like(tensor)
            
            for b in range(batch_size):
                # Apply APC to each matrix in the batch
                a1 = tensor[b].sum(-1, keepdims=True)  # Row sums
                a2 = tensor[b].sum(-2, keepdims=True)  # Column sums
                a12 = tensor[b].sum((-1, -2), keepdims=True)  # Total sum
                
                avg = a1 * a2
                avg.div_(a12)  # in-place to reduce memory
                result[b] = tensor[b] - avg
            
            return result
        
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {tensor.dim()}D tensor")

    @staticmethod
    def symmetrize_matrix(
        tensor: torch.Tensor, 
        scale: bool = True,
        dim1: int = -2,
        dim2: int = -1
    ) -> torch.Tensor:
        """
        Make a tensor symmetric along specified dimensions.
        
        Parameters:
            tensor: Input tensor to symmetrize
            scale: Whether to scale the result by 0.5 (True) or leave as sum (False)
            dim1: First dimension for transposition (default: -2)
            dim2: Second dimension for transposition (default: -1)
            
        Returns:
            Symmetrized tensor

        """
        transposed = tensor.transpose(dim1, dim2)
        if scale:
            return 0.5 * (tensor + transposed)
        else:
            return tensor + transposed



class EnzymeMSADataset(Dataset):
    """
    Dataset for enzyme active site prediction with MSA and DCA features.
    
    This dataset loads:
    - Enzyme sequences, uniprot IDs, site_labels, and site_types from CSV
    - MSA files in a3m format by uniprot ID
    - DCA features (.mat and .single files) by uniprot ID
    - Applies ESM tokenization for consistency across the project


    注意事项
    - Site label starts from 1，比如[[74], [128]]中，表示的是第74个残基和第128个残基是活性位点，而不是第75个和第129个。
    - 关于MSA，由class EnzymeMSADataset和def collate_enzyme_msa_batch处理好，然后class EvoMSATransformer对象直接用，而不是class EnzymeMSADataset和def collate_enzyme_msa_batch输出raw MSAs，再由class EvoMSATransformer对象的forward函数处理
    - class EvoMSATransformer与class EvolutionaryScaleModeling使用不同的分词器、映射表
    """
    def __init__(
        self,
        csv_file: str,
        msa_dir: str = None,
        dca_dir: str = None,
        max_seq_length: int = 1024,
        max_msa_sequences: int = 128,
        split: str = "train",
        dca_normalization: str = "minmax",
        esm_model_name: str = "esm2_t33_650M_UR50D",
        msa_model_name: str = "esm_msa1b_t12_100M_UR50S",
        use_mat_format: bool = True,
        train_num_samples: Optional[int] = None,
        test_num_samples: Optional[int] = None
    ):
        """
        Initialize the enzyme MSA dataset.
        
        Args:
            csv_file: CSV file with columns: Sequence, Uniprot ID, site_labels, site_types
            msa_dir: Directory containing MSA files (named as {uniprot_id}.a3m)
            dca_dir: Directory containing DCA output files (.mat/.single or .npy files)
            max_seq_length: Maximum sequence length
                * 对于一个序列样本，max_seq_length是允许最大的有效残基数
                * 区分'某batch内最终序列长度seq_len'和max_seq_length: 
                    * 首先'某batch内最终序列长度seq_len'是对于一个batch内部而言，经过填充后，一个batch内的所有样本序列的长度都是'某batch内最终序列长度seq_len'
                    * 即使规定了全局的max_seq_length，最终经过填充后，不同batch之间的'某batch内最终序列长度seq_len'也可能不一样
                    * 对于一个batch，填充的一般过程是：首先对于batch内的每条序列，对比'序列长度'和max_seq_length以截断过长序列，然后计算max_len = max[len(seq) for seq in seq_list]，最后确定'某batch内最终序列长度seq_len' = max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)
            max_msa_sequences: Maximum number of sequences in MSA
            split: Dataset split ("train", "val", "test")
            dca_normalization: Normalization method for DCA features
            esm_model_name: ESM model name to use for tokenization
            msa_model_name: MSA Transformer model name for MSA processing
            use_mat_format: If True, use .mat/.single files; if False, use .npy files
            train_num_samples: Number of training samples to use (None means use all samples, only applied for train split)
            test_num_samples: Number of test samples to use (None means use all samples, only applied for test split)
        """
        # Load CSV data
        self.csv_file = csv_file
        self.data = self._load_csv_data()
        
        # Store configuration
        self.msa_dir = msa_dir
        self.dca_dir = dca_dir
        self.max_seq_length = max_seq_length
        self.max_msa_sequences = max_msa_sequences
        self.split = split
        self.dca_normalization = dca_normalization
        self.use_mat_format = use_mat_format
        self.train_num_samples = train_num_samples
        self.test_num_samples = test_num_samples
        
        # Ensure directories exist
        os.makedirs(self.msa_dir, exist_ok=True)
        os.makedirs(self.dca_dir, exist_ok=True)
        
        # Load ESM alphabet for tokenization
        self.alphabet = self.get_esm_alphabet(esm_model_name)
        self.sample_converter = self.alphabet.get_sample_converter(truncation_seq_length=self.max_seq_length)   # 处理单条序列

        # Load MSA Transformer alphabet for tokenization
        self.msa_alphabet = self.get_esm_alphabet(msa_model_name)
        self.msa_alphabet.use_msa = True
        self.msa_batch_converter = self.msa_alphabet.get_batch_converter(truncation_seq_length=self.max_seq_length)
        
        # Filter valid entries - directly modify self.data to keep only valid entries
        self.data = self._filter_valid_data()
        
        # Apply train_num_samples for training split
        if self.split == "train" and self.train_num_samples is not None and self.train_num_samples > 0:
            original_size = len(self.data)
            if self.train_num_samples < original_size:
                # Randomly sample train_num_samples from the dataset
                self.data = self.data.sample(n=self.train_num_samples, random_state=42).reset_index(drop=True)
                logging.info(f"Sampled {self.train_num_samples} training samples from {original_size} total samples")
            else:
                logging.info(f"train_num_samples ({self.train_num_samples}) >= dataset size ({original_size}), using all samples")
        
        # Apply test_num_samples for test split
        if self.split == "test" and self.test_num_samples is not None and self.test_num_samples > 0:
            original_size = len(self.data)
            if self.test_num_samples < original_size:
                # Take first test_num_samples from the dataset (no randomization for test)
                self.data = self.data.head(self.test_num_samples).reset_index(drop=True)
                logging.info(f"Using first {self.test_num_samples} test samples from {original_size} total samples")
            else:
                logging.info(f"test_num_samples ({self.test_num_samples}) >= dataset size ({original_size}), using all samples")
        
        logging.info(f"Loaded {len(self.data)} valid enzyme entries for {split} split")
        logging.info(f"Using MSA model: {msa_model_name} for MSA processing")
    


    def _load_csv_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Expected CSV columns:
        - Sequence: Protein sequence
        - Uniprot ID: UniProt identifier  
        - site_labels: String representation of site labels list
        - site_types: Site type information
        
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(self.csv_file)
            
            # Check required columns
            required_cols = ['Sequence', 'Uniprot ID', 'site_labels', 'site_types']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in CSV: {missing_cols}")
            
            # Parse site_labels - handle nested list format
            def parse_site_labels(labels_str):
                if pd.isna(labels_str) or labels_str == '':
                    return []
                if isinstance(labels_str, str):
                    # Parse string representation of nested list like "[[17, 23], [42]]"
                    try:
                        return eval(labels_str)  # Use eval to parse nested list
                    except Exception as e:
                        logging.warning(f"Failed to parse site_labels: {labels_str}, error: {e}")
                        return []
                elif isinstance(labels_str, (list, tuple)):
                    return list(labels_str)
                else:
                    return []
            
            # Parse site_types
            def parse_site_types(types_str):
                if pd.isna(types_str) or types_str == '':
                    return []
                if isinstance(types_str, str):
                    try:
                        return eval(types_str)  # Use eval to parse list
                    except Exception as e:
                        logging.warning(f"Failed to parse site_types: {types_str}, error: {e}")
                        return []
                elif isinstance(types_str, (list, tuple)):
                    return list(types_str)
                else:
                    return []
            
            df['parsed_site_labels'] = df['site_labels'].apply(parse_site_labels)
            df['parsed_site_types'] = df['site_types'].apply(parse_site_types)
            
            # Filter out entries without valid sequences or site labels
            valid_mask = (
                df['Sequence'].notna() & 
                (df['Sequence'].str.len() > 0) &
                df['Uniprot ID'].notna() &
                (df['parsed_site_labels'].apply(len) > 0)
            )
            
            df = df[valid_mask].reset_index(drop=True)
            
            logging.info(f"Loaded {len(df)} valid entries from CSV file")
            return df
            
        except Exception as e:
            logging.error(f"Error loading CSV file {self.csv_file}: {e}")
            raise
    


    def get_esm_alphabet(self, model_name: str):
        """
        Create an ESM alphabet based on model name without loading the model weights.
        
        Args:
            model_name: Name of the ESM model (e.g., "esm2_t33_650M_UR50D") or Name of the ESM model
            
        Returns:
            Alphabet: The appropriate alphabet for the specified model
        """
        # Handle ESM2 models
        if model_name.startswith("esm2_"):
            # All ESM2 models use the ESM-1b alphabet
            return Alphabet.from_architecture("ESM-1b")
        
        # Handle MSA Transformer models
        elif "msa" in model_name.lower():
            return Alphabet.from_architecture("MSA Transformer")
        
        # Handle ESM-1b models
        elif model_name.startswith("esm1b_") or "esm1b" in model_name:
            return Alphabet.from_architecture("ESM-1b")
        
        # Default to ESM-1b alphabet (most commonly used)
        else:
            return Alphabet.from_architecture("ESM-1b")
    


    def _filter_valid_data(self) -> pd.DataFrame:
        """
        验证每个蛋白质条目是否具备完整的文件集合，并返回过滤后的DataFrame:
            * 序列文件：已经在CSV中
            * MSA文件：多序列比对文件
            * DCA文件：直接耦合分析文件
        
        Returns:
            Filtered DataFrame containing only valid entries
        """
        valid_rows = []
        
        for idx, row in self.data.iterrows():
            uniprot_id = row['Uniprot ID']
            
            # Check if MSA file exists
            msa_file = os.path.join(self.msa_dir, f"{uniprot_id}.a3m")
            if not os.path.exists(msa_file):
                continue
            
            # Check if DCA files exist
            if self.use_mat_format:
                # Check for .mat and .single files
                coupling_file = os.path.join(self.dca_dir, f"{uniprot_id}.mat")
                single_site_file = os.path.join(self.dca_dir, f"{uniprot_id}.single")
            else:
                # Check for .npy files
                coupling_file = os.path.join(self.dca_dir, f"{uniprot_id}_coupling_matrix.npy")
                single_site_file = os.path.join(self.dca_dir, f"{uniprot_id}_single_site_potentials.npy")
            
            if os.path.exists(coupling_file) and os.path.exists(single_site_file):
                valid_rows.append(row)
        
        # Convert list of valid rows back to DataFrame and reset index
        if valid_rows:
            valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
        else:
            # Return empty DataFrame with same columns if no valid entries
            valid_df = pd.DataFrame(columns=self.data.columns)
        
        return valid_df
    


    def parse_msa_file(self, msa_path: str, max_seqs: int = None, keep_gaps: bool = True, 
                    keep_insertions: bool = False, to_upper: bool = False) -> List[Tuple[str, str]]:
        """
        Parse MSA file.

        确保加载的MSA文件内部对齐
        
        Args:
            msa_path: Path to MSA file (a3m format)
            max_seqs: Maximum number of sequences to return
            keep_gaps: Whether to keep gap characters ('-')
            keep_insertions: Whether to keep lowercase letters (insertions in a3m)
            to_upper: Whether to convert sequences to uppercase
        
        Returns:
            a list of sequences from the MSA file, be like:
                [
                    (label1, sequence1),  # 序列1 - 通常是查询序列
                    (label2, sequence2),  # 序列2
                    (label3, sequence3),  # 序列3
                    ...
                ]
        """
        if not msa_path or not os.path.exists(msa_path):
            return []
            
        try:
            f = open(msa_path, 'r')
                
            msa_seqs = []
            seq = ""
            header = ""
            
            # Parse MSA file line by line
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if seq:
                        msa_seqs.append((header, seq))
                        if max_seqs and len(msa_seqs) >= max_seqs:
                            break
                    header = line[1:]
                    seq = ""
                else:
                    # Process sequence based on parameters
                    if not keep_insertions:
                        # More efficient than regex for simple character removal
                        deletekeys = dict.fromkeys(string.ascii_lowercase)
                        line = line.translate(str.maketrans('', '', ''.join(deletekeys)))
                    if not keep_gaps:
                        line = line.replace('-', '')
                    if to_upper:
                        line = line.upper()
                    seq += line
            
            
            if seq:         # Add the last sequence
                msa_seqs.append((header, seq))

            f.close()

            if msa_seqs:            # Consistency check for sequence lengths (MSA should be aligned)
                seq_lengths = set(len(seq) for _, seq in msa_seqs)
                if len(seq_lengths) > 1:
                    logging.warning(f"MSA sequences have different lengths: {seq_lengths}")
            
            return msa_seqs[:max_seqs]      # Return (label, seq) pairs, limited by max_seqs
            
        except Exception as e:
            logging.error(f"Error parsing MSA file {msa_path}: {e}")
            return []
        
    
    
    def parse_and_tokenize_msa(self, msa_path: str, max_tokens_seqs: int = None) -> Tuple[List[Tuple[str, str]], torch.Tensor]:
        """
        处理 raw MSA 文件
            * 调用self.msa_batch_converter，将raw MSA前 max_tokens_seqs个序列转成msa_tokens
            * 调用self.conservation, 处理raw MSA所有序列转成保守性得分
        
        Args:
            msa_path: MSA file, processed by parse_msa_file, a list of sequences from the MSA file, be like:
                [
                    (label1, sequence1),  # 序列1 - 通常是查询序列
                    (label2, sequence2),  # 序列2
                    (label3, sequence3),  # 序列3
                    ...
                ]
            max_tokens_seqs: Maximum number of sequences to include in tokenized output
            
        Returns:
            Tuple containing:
                - conservation_scores: 1D torch.FloatTensor, [valid_len]
                - msa_tokens: Tensor of tokenized MSA sequences [<=max_tokens_seqs, seq_len_i]
        """
        all_msa_seqs = self.parse_msa_file(msa_path, max_seqs=None, keep_gaps=True)
        
        # 对于tokenization，仅使用指定数量的序列
        if max_tokens_seqs is not None and len(all_msa_seqs) > max_tokens_seqs:
            tokenization_seqs = all_msa_seqs[:max_tokens_seqs]
        else:
            tokenization_seqs = all_msa_seqs
        
        # 使用MSA Transformer的批处理转换器进行tokenization
        '''
        msa_tokens: [batch_size, <=max_seqs, seq_len_i]
            * 这里是处理一个样本的MSA，所以batch_size==1
            * <=max_seqs表示一个样本的MSA深度可能小于max_seqs，而执行下面的self.msa_batch_converter并没有对单样本MSA深度进行填充
            * seq_len_i的含义是：一个样本对应MSA的第一条序列（即该样本序列）经过截断、映射、填充self.alphabet.prepend_bos和self.alphabet.append_eos得到的
                长度，seq_len_i是一个样本对应的MSA文件内部对齐长度，在一个batch中，可能样本之间的MSA对齐长度不一样，所以用seq_len_i表示。
            * 先对齐单个样本MSA的长度seq_len_i，在批包装时，再对齐一个batch中所有样本的MSA长度seq_len
        '''
        _, _, msa_tokens = self.msa_batch_converter([tokenization_seqs])

        # 使用所有可用的MSA序列计算保守性得分
        conservation_scores = self.conservation(all_msa_seqs)
        
        # 返回处理后的张量，移除批次维度，得到[<=max_seqs, seq_len_i]
        return conservation_scores, msa_tokens.squeeze(0)
    

    
    def load_dca_features_mat(self, coupling_path: str, single_site_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load DCA features from .mat and .single files.
        
        Args:
            coupling_path: Path to coupling matrix file (.mat)
            single_site_path: Path to single-site potentials file (.single)
            
        Returns:
            Tuple of (coupling_matrix, single_site_potentials)
                coupling_matrix：[valid_len, valid_len]
                single_site_potentials：[valid_len, 20]
        """
        try:
            # Load coupling matrix from .mat file
            with open(coupling_path, 'r') as f:
                lines = []
                for line in f:
                    # Strip whitespace and trailing tabs, then split by tabs
                    cleaned_line = line.strip().rstrip('\t')
                    if cleaned_line:  # Skip empty lines
                        values = [float(x) for x in cleaned_line.split('\t') if x.strip()]
                        lines.append(values)
            
            # shape: [len(sequence), valid_len]
            coupling_matrix = torch.tensor(lines, dtype=torch.float)
            
            # Load single-site potentials from .single file
            with open(single_site_path, 'r') as f:
                lines = []
                for line in f:
                    # Strip whitespace and trailing tabs, then split by tabs
                    cleaned_line = line.strip().rstrip('\t')
                    if cleaned_line:  # Skip empty lines
                        values = [float(x) for x in cleaned_line.split('\t') if x.strip()]
                        if len(values) == 20:  # Should have exactly 20 amino acid values
                            lines.append(values)
                        else:
                            logging.warning(f"Line with {len(values)} values instead of 20 in {single_site_path}")
            
            # shape: [len(sequence), 20]
            single_site_potentials = torch.tensor(lines, dtype=torch.float)

            # truncation
            if hasattr(self, 'max_seq_length') and self.max_seq_length > 0:
                effective_seq_length = self.max_seq_length               
                if coupling_matrix.size(0) > effective_seq_length:
                    coupling_matrix = coupling_matrix[:effective_seq_length, :effective_seq_length]
                    logging.info(f"Truncated coupling_matrix to {effective_seq_length}x{effective_seq_length}")               
                if single_site_potentials.size(0) > effective_seq_length:
                    single_site_potentials = single_site_potentials[:effective_seq_length, :]
                    logging.info(f"Truncated single_site_potentials to {effective_seq_length}x20")
            
            return coupling_matrix, single_site_potentials
            
        except Exception as e:
            logging.error(f"Error loading DCA features from .mat/.single files: {e}")
            raise
    


    def load_dca_features_npy(self, coupling_path: str, single_site_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注意本函数未经过审查！！！
        Load DCA features from .npy files.
        
        Args:
            coupling_path: Path to coupling matrix file (.npy)
            single_site_path: Path to single-site potentials file (.npy)
            
        Returns:
            Tuple of (coupling_matrix, single_site_potentials)
        """
        try:
            coupling_matrix = torch.from_numpy(np.load(coupling_path)).float()
            single_site_potentials = torch.from_numpy(np.load(single_site_path)).float()
            
            return coupling_matrix, single_site_potentials
            
        except Exception as e:
            logging.error(f"Error loading DCA features from .npy files: {e}")
            raise
    


    def conservation(self, msa_seqs: List[Tuple[str, str]], gap_threshold: float = 0.5, normalize: bool = True) -> torch.FloatTensor:

        """
        为每个残基计算保守性得分（标量值）。

        Compute conservation scores using Henikoff-style weights + Jensen-Shannon Divergence (Capra07).
            – Frequency estimate: vectorized Henikoff weights (O(N·L))
            – Divergence: JSD between weighted freq p and background q
            – Gap handling: positions with gap_fraction >= threshold set to mean−std
            – Normalization: optional Z-score

        * 借鉴于https://github.com/TheApacheCats/al2co

        * 关于从al2co各种方法组合中选出适合预测活性位点任务的论述：https://chatgpt.com/s/t_686fb747545881918938ba9f6172f584, https://chatgpt.com/s/t_686fc15a718481919954959a4eacda04
        
        Args:
            msa_seqs: parse_msa_file函数返回的(label, sequence)元组列表
            gap_threshold: 空位比例阈值，高于此阈值的位置将特殊处理
            normalize: 是否执行Z-score标准化
            
        Returns:
            形状为[L]的保守性得分张量，L为比对长度
        """
        # extract sequences and basic dims
        sequences = [seq for _, seq in msa_seqs]
        if not sequences:
            return torch.tensor([])
        N = len(sequences)
        L = len(sequences[0])

        # amino acids and background freq (SwissProt 2025_03)
        AA = "ACDEFGHIKLMNPQRSTVWY"
        BG_FREQ = torch.tensor([
            0.0825,0.0138,0.0546,0.0671,0.0386,
            0.0707,0.0227,0.0590,0.0579,0.0964,
            0.0241,0.0406,0.0474,0.0393,0.0552,
            0.0665,0.0536,0.0685,0.0110,0.0292
        ], dtype=torch.float32, device='cpu')

        # build token matrix [N, L], invalid positions = -1
        msa_tokens = torch.full((N, L), -1, dtype=torch.long)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                idx = AA.find(aa.upper())
                if idx >= 0:
                    msa_tokens[i, j] = idx

        # valid mask and gap fractions
        valid = (msa_tokens >= 0)
        gap_frac = (~valid).float().mean(dim=0)  # [L]

        # one-hot [N, L, 20]
        one_hot = F.one_hot(msa_tokens.clamp(min=0), num_classes=20).float()
        one_hot = one_hot * valid.unsqueeze(-1).float()  # mask gaps out

        # compute per-column amino acid counts and distinct types
        counts = one_hot.sum(dim=0)                # [L,20]
        types = (counts > 0).sum(dim=1)            # [L]

        # vectorized Henikoff weights:
        #  for each (seq,pos): weight = valid / (count[aa,pos] * types[pos])
        #  then normalize per column
        # gather count_sp: [N,L] = counts[pos, aa]  
        #  note: transpose counts to [20,L] to index by msa_tokens
        counts_T = counts.transpose(0, 1)          # [20, L]
        pos_idx = torch.arange(L)
        # index counts and types
        count_sp = counts_T[msa_tokens.clamp(min=0), pos_idx.unsqueeze(0)]  # [N, L]
        types_p = types.unsqueeze(0).expand(N, -1).float()                  # [N, L]

        # base weights: 1 / (count * types), zero where invalid or gap
        base = count_sp * types_p
        base = base.clamp(min=1e-12)
        weights = valid.float() / base

        # normalize per column so sum_i weights[i,pos] == 1
        col_sum = weights.sum(dim=0, keepdim=True)        # [1, L]
        weights = weights / col_sum

        # compute weighted frequencies p [L,20]
        weighted_one_hot = one_hot * weights.unsqueeze(-1)  # [N, L, 20]
        token_counts = weighted_one_hot.sum(dim=0)          # [L, 20]
        seq_counts = weights.sum(dim=0).unsqueeze(-1)       # [L, 1]
        seq_counts = seq_counts.clamp(min=1e-12)
        p = token_counts / seq_counts                       # [L,20]

        # background q
        q = BG_FREQ.unsqueeze(0).expand(L, -1)              # [L,20]

        # Jensen-Shannon Divergence
        m = 0.5 * (p + q)
        eps = 1e-12
        kl_pm = (p * (p.div(m).clamp(min=eps).log())).sum(dim=1)
        kl_qm = (q * (q.div(m).clamp(min=eps).log())).sum(dim=1)
        jsd = 0.5 * (kl_pm + kl_qm)                         # [L]

        # gap handling: set gappy columns to mean−std of valid
        mask_ok = gap_frac < gap_threshold
        if mask_ok.sum() > 0:
            valid_scores = jsd[mask_ok]
            mu, sigma = valid_scores.mean(), valid_scores.std()
            jsd[~mask_ok] = mu - sigma
            # optional Z-score normalization
            if normalize and sigma > 0:
                jsd = (jsd - mu) / sigma

        return jsd



    def _create_active_site_labels(self, sequence_length: int, site_labels: List[List[int]], site_types: List[int] = None) -> torch.Tensor:
        """
        Create active site labels tensor from site labels and types.
        Note: Site label starts from 1，比如[[74], [128]]中，表示的是第74个残基和第128个残基是活性位点，而不是第75个和第129个。
        
        labels
        Args:
            sequence_length: Length of the sequence
            site_labels: List of site label lists, e.g., [[17, 23], [42], [74]]
            site_types: List of site types corresponding to each site label, e.g., [1, 1, 2]
            
        Returns:
            Tensor of active site labels [effective_seq_len]
                * 其中effective_seq_len =  min(len(original_sequence), self.max_seq_length)
                * 0表示非活性位点，非0表示活性位点，更进一步1表示结合位点，2表示催化位点，3表示其他位点
                * labels中不包含special token(cls/bos, eos, pad)
        """
        labels = torch.zeros(sequence_length, dtype=torch.long)
        
        # Use default site type 1 if not provided
        if site_types is None:
            raise ValueError(
                "site_types parameter is required and cannot be None. "
                "Please provide a list of site types corresponding to each site label in "
                f"function '_create_active_site_labels'. Expected {len(site_labels)} site types."
            )
        
        for one_site, site_type in zip(site_labels, site_types):
            if len(one_site) == 1:
                # Single position (e.g., [74])
                idx = one_site[0] - 1  # Convert to 0-indexed
                if 0 <= idx < sequence_length:
                    labels[idx] = site_type
            elif len(one_site) == 2:
                # Range of positions (e.g., [17, 23])
                start, end = one_site[0] - 1, one_site[1] - 1  # Convert to 0-indexed
                for idx in range(start, end + 1):
                    if 0 <= idx < sequence_length:
                        labels[idx] = site_type
            else:
                logging.warning(f"Unexpected site label format: {one_site}")
        
        return labels
    

    
    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        return len(self.data)
    


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        约定各种长度的含义：
            * seq_len：某个batch内经过填充后每条序列的长度一致，都为seq_len，包含cls/bos, eos, pad
            * valid_len: min(len(sequence), self.max_seq_length)，原始序列长度len(sequence)，如果超过self.max_seq_length,则取self.max_seq_length，总之valid_len表示全都是有效残基，不包括special token
            * len(sequence)：原始序列长度
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
            - sequence: seq_tokens，一维张量，长度valid_len，原始序列经过截断、氨基酸字母到数字映射，但不进行special token填充
            - msa: msa_tokens，单个样本MSA tokens[<=max_seqs, seq_len_i]，seq_len_i是一个样本对应的MSA文件内部对齐长度
            - coupling_matrix: [valid_len, valid_len]
            - single_site_potentials: DCA single-site potentials [valid_len, 20]
            - conservation_scores: 保守性得分 [valid_len]
            - labels: Active site labels [valid_len]
            - uniprot_id: UniProt identifier
            - sequence_length: Original sequence length
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        row = self.data.iloc[idx]
        
        uniprot_id = row['Uniprot ID']
        sequence = row['Sequence']
        site_labels = row['parsed_site_labels']
        site_types = row['parsed_site_types']

        # Convert sequence to tokens
        _, seq_tokens = self.sample_converter(sequence)

        # Process MSA file and get conservation scores and tokens
        msa_file = os.path.join(self.msa_dir, f"{uniprot_id}.a3m")
        conservation_scores, msa_tokens = self.parse_and_tokenize_msa(
            msa_file, max_tokens_seqs=self.max_msa_sequences
        )

        # load DCA features
        if self.use_mat_format:
            coupling_file = os.path.join(self.dca_dir, f"{uniprot_id}.mat")
            single_site_file = os.path.join(self.dca_dir, f"{uniprot_id}.single")
            coupling_matrix, single_site_potentials = self.load_dca_features_mat(coupling_file, single_site_file)
        else:
            coupling_file = os.path.join(self.dca_dir, f"{uniprot_id}_coupling_matrix.npy")
            single_site_file = os.path.join(self.dca_dir, f"{uniprot_id}_single_site_potentials.npy")
            coupling_matrix, single_site_potentials = self.load_dca_features_npy(coupling_file, single_site_file)


        # Calculate effective sequence length after potential truncation
        effective_seq_len = min(len(sequence), self.max_seq_length)

        # Create active site labels
        labels = self._create_active_site_labels(effective_seq_len, site_labels, site_types)

        # Ensure conservation_scores length matches the effective sequence length
        if len(conservation_scores) > effective_seq_len:
            conservation_scores = conservation_scores[:effective_seq_len]

        return {
            'sequence': seq_tokens,
            'msa': msa_tokens,
            'coupling_matrix': coupling_matrix,
            'single_site_potentials': single_site_potentials,
            'conservation_scores': conservation_scores,
            'labels': labels,
            'uniprot_id': uniprot_id,
            'sequence_length': len(sequence)
        }
    


def collate_enzyme_msa_batch(
    batch: List[Dict[str, Union[torch.Tensor, List[torch.Tensor], str, int]]],
) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], List[str], List[int]]]:
    """
    Collate function for EnzymeMSADataset batches.

    特别注意labels张量！！！！！！！！！！！！！！！！！！！
    labels = torch.zeros(batch_size, max_seq_len, dtype=torch.float)
        * labels张量与sequences张量形状一致，包含special token位置的标签(值为0)
        * 在计算损失前可能需要通过attention mask或其他方式忽略special token位置
    
    Args:
        batch: List of samples from EnzymeMSADataset
        
    Returns:
        Batched dictionary with padded tensors for sequences, MSA, and labels,
        with proper special token handling based on ESM conventions
    """
    # ESM-1b style token configuration
    prepend_bos = True
    append_eos = True
    padding_idx = 1  # <pad> token index
    cls_idx = 0      # <cls> token index (used as BOS)
    eos_idx = 2      # <eos> token index
    
    # Get batch size
    batch_size = len(batch)
    
    # Get maximum content lengths (without special tokens)
    max_seq_content_len = max(sample['sequence'].size(0) for sample in batch)
    
    # Calculate total sequence length including special tokens
    max_seq_len = max_seq_content_len + int(prepend_bos) + int(append_eos)
    

    # Get MSA dimensions，MSA tokens现在直接是张量[num_seqs, seq_len_i]
    # 注意这里的seq_len_i已经包含了cls/bos, eos, pad(即单个样本MSA内部已经对齐)，在collate_enzyme_msa_batch中处理MSAs的目的是对齐样本之间的长度，只需继续填充pad，无需填充cls/bos, eos
    max_msa_seqs = max(sample['msa'].size(0) for sample in batch if sample['msa'].size(0) > 0)
    if max_msa_seqs == 0:
        max_msa_seqs = 1  # 确保至少有一个序列
    max_msa_len = max(sample['msa'].size(1) for sample in batch if sample['msa'].size(0) > 0)
    if max_msa_len == 0:
        max_msa_len = 1  # 确保至少有一个位置


    # Initialize batch tensors with padding_idx instead of zeros
    # MSA Transformer的padding_idx和ESM2的padding_idx相同，都是1，所以这里msa_tokens = torch.full直接用ESM2的padding_idx填充
    sequences = torch.full((batch_size, max_seq_len), padding_idx, dtype=torch.long)
    msa_tokens = torch.full((batch_size, max_msa_seqs, max_msa_len), padding_idx, dtype=torch.long)
    
    # Other fields
    coupling_matrices = []
    single_site_potentials = []

    # Labels with same shape as sequences - including positions for special tokens
    # All special token positions will have label 0
    labels = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

    conservation_scores = torch.zeros(batch_size, max_seq_len, dtype=torch.float)
    uniprot_ids = []
    sequence_lengths = []
    
    # Process each sample in the batch
    # 该循环确保batch内顺序：sequences[i] ↔ coupling_matrices[i] ↔ single_site_potentials[i] ↔ label[i] ↔ uniprot_ids[i]
    for i, sample in enumerate(batch):      
        # Handle sequence with special tokens
        seq = sample['sequence']
        seq_len = seq.size(0)
        
        # Add BOS/CLS token if configured
        if prepend_bos:
            sequences[i, 0] = cls_idx
        
        # Place content tokens after BOS (if any)
        sequences[i, int(prepend_bos):seq_len + int(prepend_bos)] = seq

        label_len = sample['labels'].size(0)
        labels[i, int(prepend_bos):label_len + int(prepend_bos)] = sample['labels']

        # Add EOS token if configured
        if append_eos:
            sequences[i, seq_len + int(prepend_bos)] = eos_idx
        
        # Handle MSA padding (现在是直接填充整个张量)
        # 如果一个样本的msa深度不足max_msa_seqs，空缺的序列全都设置为pad
        msa = sample['msa']
        if msa.size(0) > 0:  # 确保有序列
            num_seqs = min(msa.size(0), max_msa_seqs)   # 比较robust的写法，实际上不用这么写
            seq_len = min(msa.size(1), max_msa_len)     # 比较robust的写法，实际上不用这么写
            msa_tokens[i, :num_seqs, :seq_len] = msa[:num_seqs, :seq_len]
        
        # Store coupling matrices directly without padding
        coupling_matrices.append(sample['coupling_matrix'])
        
        # Store single-site potentials directly without padding
        single_site_potentials.append(sample['single_site_potentials'])

        # 处理保守性得分 - 与序列内容对齐
        cons_len = sample['conservation_scores'].size(0)
        conservation_scores[i, int(prepend_bos):cons_len + int(prepend_bos)] = sample['conservation_scores']
        
        # Collect metadata
        uniprot_ids.append(sample['uniprot_id'])
        sequence_lengths.append(sample['sequence_length'])
    
    return {
        'sequences': sequences,                                 # 2维张量：[batch_size, max_seq_len]
        'msa': msa_tokens,                                      # 3维张量：[batch_size, max_msa_seqs, max_msa_len]
        'coupling_matrices': coupling_matrices,                 # 2维张量列表：[coupling_matrix_0, coupling_matrix_1,coupling_matrix_2,...]，其中coupling_matrix_i形状为：[valid_len_i, valid_len_i]
        'single_site_potentials': single_site_potentials,       # 2维张量列表：[single_site_potential_0, single_site_potential_1,single_site_potential_2,...]，其中single_site_potential_i形状为：[valid_len_i, 20]
        'conservation_scores': conservation_scores,             # [batch_size, max_seq_len]
        'labels': labels,                                       # 2维张量：[batch_size, max_seq_len]，与sequences形状一致
        'uniprot_ids': uniprot_ids,                             # 字符串列表
        'sequence_lengths': sequence_lengths                    # 整数列表
    }  



def create_enzyme_msa_dataloader(
    dataset: EnzymeMSADataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 16,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader for the EnzymeMSADataset.
    
    Args:
        dataset: The EnzymeMSADataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # 推荐设置为CPU核心数
        pin_memory=pin_memory,
        collate_fn=collate_enzyme_msa_batch,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=drop_last
    )



atom_types = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
    'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta',
    'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs'
]



def pad_atom_distance_matrix(adm_list):
    max_size = max([adm.shape[0] for adm in adm_list])
    adm_list = [

        # function: numpy.lib.arraypad.pad
        torch.tensor(np.pad(adm, (0, max_size - adm.shape[0]),
                            'maximum')).unsqueeze(0).long() for adm in adm_list
    ]
    return torch.cat(adm_list, dim=0)



def _calculate_rxn_hash(rxn: str) -> str:
    """Computes a SHA256 hash for a given reaction string to create a unique identifier."""
    hash_object = hashlib.sha256()
    hash_object.update(rxn.encode("utf-8"))
    return hash_object.hexdigest()



def get_adm(mol, max_distance=6):
    """
    The get_adm function is used to create an Atom Distance Matrix (ADM) for a molecule, which represents the shortest path distance between each pair of atoms in the molecule.
    """
    mol_size = mol.GetNumAtoms()
    distance_matrix = np.ones((mol_size, mol_size)) * max_distance + 1

    # [function: rdkit.Chem.rdmolops.GetDistanceMatrix] returns the topological distance matrix of all atom pairs in the molecule, which represents the number of bonds in the shortest path between two atoms (without considering geometric distances).
    dm = Chem.GetDistanceMatrix(mol)
    dm[dm > 100] = -1  # remote (different molecule)
    dm[dm > max_distance] = max_distance  # remote (same molecule)
    dm[dm == -1] = max_distance + 1
    distance_matrix[:dm.shape[0], :dm.shape[1]] = dm
    return distance_matrix



class ReactionFeatures(object):
    def __init__(self, reaction_features, rxn=None) -> None:

        (self.react_fgraph, self.react_dgraph), (self.prod_fgraph, self.prod_dgraph) = (
            reaction_features
        )

        self.react_dgraph = torch.from_numpy(self.react_dgraph)
        self.prod_dgraph = torch.from_numpy(self.prod_dgraph)
        self.rxn = rxn

    def __repr__(self) -> str:
        repr_str = f"Reaction(reactants(fgraph:{self.react_fgraph},dgraph:{self.react_dgraph}; products(fgraph:{self.prod_fgraph}, dgraph:{self.prod_dgraph}) )"
        return repr_str

    def to(self, device):
        self.react_fgraph = self.react_fgraph.to(device)
        self.react_dgraph = self.react_dgraph.to(device)
        self.prod_fgraph = self.prod_fgraph.to(device)
        self.prod_dgraph = self.prod_dgraph.to(device)



def get_structure_sequence(pdb_file):
    try:
        # [function: rdkit.Chem.rdmolfiles.MolFromPDBFile] returns [class: rdkit.Chem.rdchem.Mol] object.
        mol = Chem.MolFromPDBFile(pdb_file)

        # function: rdkit.Chem.rdmolfiles.MolToSequence
        protein_sequence = Chem.MolToSequence(mol)
    except:
        protein_sequence = ''
    return protein_sequence



def multiprocess_structure_check(df, nb_workers):
    if nb_workers != 0:

        pandarallel.initialize(nb_workers=nb_workers, progress_bar=False)
        df['Sequence_calculated'] = df['pdb_files'].parallel_apply(
            lambda x: get_structure_sequence(x))
    else:
        top_tqdm.pandas(desc='pandas bar')
        df['Sequence_calculated'] = df['pdb_files'].progress_apply(
            lambda x: get_structure_sequence(x))

    return df



class NotCanonicalizableSmilesException(ValueError):
    pass



def canonicalize_smiles(smi, remove_atom_mapping=False, erro_raise=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        if remove_atom_mapping:
            [
                atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()
                if atom.HasProp('molAtomMapNumber')
            ]
        return Chem.MolToSmiles(mol)
    else:
        if erro_raise:
            raise NotCanonicalizableSmilesException(
                'Molecule not canonicalizable')
        return ''
    


def process_reaction(rxn):
    reactants, reagents, products = rxn.split('>')
    try:
        precursors = [
            canonicalize_smiles(r, True, True) for r in reactants.split(".")
        ]
        if len(reagents) > 0:
            precursors += [
                canonicalize_smiles(r, True, True) for r in reagents.split(".")
            ]
        products = [
            canonicalize_smiles(p, True, True) for p in products.split(".")
        ]
    except NotCanonicalizableSmilesException:
        return ""

    joined_precursors = canonicalize_smiles(".".join(sorted(precursors)))
    joined_products = canonicalize_smiles(".".join(sorted(products)))

    return f"{joined_precursors}>>{joined_products}"



def multiprocess_reaction_check(df, nb_workers):
    if nb_workers != 0:

        pandarallel.initialize(nb_workers=nb_workers, progress_bar=False)
        df["canonicalize_rxn_smiles"] = df["rxn_smiles"].parallel_apply(
            lambda x: process_reaction(x)
        )
    else:
        top_tqdm.pandas(desc="pandas bar")
        df["canonicalize_rxn_smiles"] = df["rxn_smiles"].progress_apply(
            lambda x: process_reaction(x)
        )

    return df



class EnzymeMSAStructureDataset(Dataset):
    """
    Dataset integrating protein sequence, structure, reaction, MSA, and DCA features for enzyme active site prediction.

    expected dataset format:
        * CSV file with columns: 'Uniprot ID', 'rxn_smiles_single', 'Sequence', 'ec', 'site_labels', 'site_types'.
            * 'Uniprot ID' be like L0E4H0
            * 'rxn_smiles_single' be like Cc1ccccc1.Cc1ccccc1>>Cc1ccccc(C(=O)O)c1, not include enzyme amino acid sequence between the reactants and products.
            * 'Sequence' be like MSRQLSRARPATVLGAMEMGRRMDAPTSAAVTRAFLERGHTEIDTAFVYSEGQSETILGGLGL
            * 'site_labels' be like [[17, 23], [42], [74], [172], [168]], where each sublist represents a range or single residue index of active sites. Site label starts from 1，比如[[74], [128]]中，表示的是第74个残基和第128个残基是活性位点，而不是第75个和第129个。
            * 'site_types' be like [1, 1, 1, 1, 2], where 1 indicates binding site and 2 indicates catalytic site and 3 indicates other site, 0 indicates non-active site.
            * 'ec' be like 1.1.1.104 or 1.1.-.-, where - indicates unknown level.
        * Protein structures in PDB format located in `structure_path`.
        * MSA files in .a3m format located in `msa_dir`, named as {uniprot_id}.a3m.
        * DCA files in .mat and .single format located in `dca_dir`, named as {uniprot_id}.mat and {uniprot_id}.single.
    """
    def __init__(
        self,
        csv_file: str,
        structure_path: str,
        msa_dir: str,
        dca_dir: str,
        dssp_exec_path: str,
        dssp_dir: str,
        graphec_radius: float = 10.0,
        protein_max_length: int = 600,
        max_msa_sequences: int = 128,
        split: str = "train",
        num_samples: Optional[int] = None,
        num_workers: int = 16,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        msa_model_name: str = "esm_msa1b_t12_100M_UR50S",
        preprocessing: Optional[str] = None,
        preprocessing_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            csv_file (str): Path to the main CSV file containing metadata.
                            Expected columns: 'Uniprot ID', 'rxn_smiles_single', 'Sequence', 'site_labels', 'site_types'.
            structure_path (str): Directory containing protein structures (e.g., alphafolddb_download).
            msa_dir (str): Directory containing MSA files in .a3m format, named as {uniprot_id}.a3m.
            dca_dir (str): Directory containing DCA files (.mat, .single), named as {uniprot_id}.mat/single.
            dssp_exec_path (str): Path to the mkdssp executable (e.g., './Features/dssp-2.0.4/').
            dssp_dir (str): Directory to save intermediate and final DSSP tensor files.
            graphec_radius (float): Radius for graph construction in geometric feature calculation. Only used in GraphEC.
            protein_max_length (int): Maximum sequence length for proteins. Longer sequences will be truncated.
            max_msa_sequences (int): Maximum number of sequences to use from each MSA.
            split: Dataset split ("train", "val", "test")
            num_samples: Number of train/test samples to use (None means use all samples)
            preprocessing (str, optional): Controls the data processing workflow. must be one of: [None, 'save', 'load']
                * None (default): Process data dynamically for each __getitem__ call.
                * 'save': Process all data once and save to disk. After saving, the object's mode will automatically switch to 'load' for immediate use. 提醒：选择save模型时，最好num_samples=None，确保所有样本都被预处理和保存。
                * 'load': Load pre-processed data from disk.
            preprocessing_dir (str, optional): Directory to save/load preprocessed data. Required when preprocessing is 'save' or 'load'.
            **kwargs: Additional keyword arguments for torchdrug.data.Protein.from_pdb.
        """
        # Initialize paths and parameters
        self.csv_file = csv_file
        self.structure_path = structure_path
        self.msa_dir = msa_dir
        self.dca_dir = dca_dir
        self.dssp_exec_path = dssp_exec_path
        self.dssp_dir = dssp_dir
        self.graphec_radius = graphec_radius
        self.protein_max_length = protein_max_length
        self.max_msa_sequences = max_msa_sequences
        self.split = split
        self.num_samples = num_samples
        self.nb_workers = num_workers
        self.preprocessing = preprocessing
        self.preprocessing_dir = preprocessing_dir
        self.kwargs = kwargs
        if not hasattr(self, 'protein_max_length'):
            self.protein_max_length = 600
        if self.preprocessing_dir:
            os.makedirs(self.preprocessing_dir, exist_ok=True)
            self.dataframe_check_cache_file = os.path.join(self.preprocessing_dir, "dataframe_check_cache.pkl")

        # Initialize featurizers for reactions 
        self.mol_to_graph = partial(mol_to_bigraph, add_self_loop=True)
        self.node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

        # Load ESM alphabet for tokenization
        self.alphabet = self.get_esm_alphabet(esm_model_name)
        self.sample_converter = self.alphabet.get_sample_converter(truncation_seq_length=self.protein_max_length)   # 处理单条序列

        # Load MSA Transformer alphabet for tokenization
        self.msa_alphabet = self.get_esm_alphabet(msa_model_name)
        self.msa_alphabet.use_msa = True
        self.msa_batch_converter = self.msa_alphabet.get_batch_converter(truncation_seq_length=self.protein_max_length)

        # Load and filter data
        self.dataset_df = self._load_and_filter_data()
        logging.info(f"Loaded {len(self.dataset_df)} valid samples from {self.csv_file}")

        # sub-sampling
        if self.num_samples is not None and self.num_samples > 0:
            full_size = len(self.dataset_df)
            if self.num_samples < full_size:
                self.dataset_df = self.dataset_df.sample(n=self.num_samples, random_state=42).reset_index(drop=True)
                logging.info(f"Sampled {self.num_samples} training samples from {full_size} total samples")
            else:
                raise ValueError(f"sub-sampling num_samples ({self.num_samples}) >= dataset size ({full_size})")
        
        self._setup_protein_transforms()

        if self.preprocessing == 'save':
            self.save_processed()
            logging.info("Preprocessing complete. All items have been saved.")



    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        switcher method based on the preprocessing mode.
        """
        if self.preprocessing == 'load':
            row = self.dataset_df.iloc[idx]
            uniprot_id = row['Uniprot ID']
            current_rxn_smiles = row['canonicalize_rxn_smiles']
            current_rxn_hash = _calculate_rxn_hash(current_rxn_smiles)
            new_filename = f"{uniprot_id}_{current_rxn_hash}.pkl"
            new_item_path = os.path.join(self.preprocessing_dir, new_filename)

            # legacy filename format
            old_filename = f"{uniprot_id}.pkl"
            old_item_path = os.path.join(self.preprocessing_dir, old_filename)


            if os.path.exists(new_item_path):
                return torch.load(new_item_path)
            
            elif os.path.exists(old_item_path):
                item = torch.load(old_item_path)
                # loaded_rxn_smiles = None
                # if 'reaction_graph' in item and hasattr(item['reaction_graph'], 'rxn'):
                #     loaded_rxn_smiles = item['reaction_graph'].rxn
                # if loaded_rxn_smiles is not None and loaded_rxn_smiles != current_rxn_smiles:
                #     logging.info(f"Notice: For Uniprot ID {uniprot_id}, the legacy file '{old_filename}' was loaded. The reaction in the file ('{loaded_rxn_smiles}') does NOT match the expected reaction ('{current_rxn_smiles}'). ")
                # elif loaded_rxn_smiles is not None and loaded_rxn_smiles == current_rxn_smiles:
                #     logging.info(f"Notice: For Uniprot ID {uniprot_id}, the legacy file '{old_filename}' was loaded. The reaction in the file ('{loaded_rxn_smiles}') does match the expected reaction ('{current_rxn_smiles}'). ")
                # else:
                #     logging.info(f"Notice: For Uniprot ID {uniprot_id}, the legacy file '{old_filename}' was loaded. The reaction in the file is unavailable, so no comparison was made.")
                return item
            else:
                raise FileNotFoundError(
                    f"Processed file not found for sample index {idx} (Uniprot ID: {uniprot_id}). "
                    f"Checked for both new format ('{new_item_path}') and legacy format ('{old_item_path}'). "
                    f"Please run the dataset with preprocessing='save' first."
                )
                
                # # 文件不存在时，不再抛出异常，而是记录一个警告并返回 None
                # row = self.dataset_df.iloc[idx]
                # uniprot_id = row['Uniprot ID']
                # logging.warning(
                #     f"Processed file not found for sample index {idx} (Uniprot ID: {uniprot_id}). "
                #     f"Skipping this sample. Checked for: '{new_item_path}' and '{old_item_path}'."
                # )
                # return None
        else:
            # Mode 'None' or 'save': Process the item on-the-fly.
            return self.getitem(idx)



    def getitem(self, idx: int) -> Dict[str, any]:
        """
        original __getitem__ method.
        """
        if idx >= len(self.dataset_df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.dataset_df)}")
        row = self.dataset_df.iloc[idx]
        uniprot_id = row['Uniprot ID']
        sequence = row['Sequence']
        rxn_smiles = row['canonicalize_rxn_smiles']
        pdb_file = row['pdb_files']
        msa_file = os.path.join(self.msa_dir, f"{uniprot_id}.a3m")
        coupling_file = os.path.join(self.dca_dir, f"{uniprot_id}.mat")
        single_site_file = os.path.join(self.dca_dir, f"{uniprot_id}.single")
        site_label = row['parsed_site_labels']
        site_type = row['parsed_site_types']
        effective_seq_len = min(len(sequence), self.protein_max_length)

        # =========================================== Top@Sequence + Structure ====================================================== 
        protein = data.Protein.from_pdb(pdb_file, self.kwargs)
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        # =========================================== Bottom@Sequence + Structure =================================================== 


        # =========================================== Top@Reaction ====================================================== 
        reaction_features = self._calculate_rxn_features(rxn_smiles)
        rxn_fclass = ReactionFeatures(reaction_features, rxn_smiles)
        # =========================================== Bottom@Reaction =================================================== 


        # =========================================== Top@MSA ======================================================
        conservation_score, msa_tokens = self.parse_and_tokenize_msa(msa_file, effective_seq_len, max_tokens_seqs=self.max_msa_sequences)
        coupling_matrix, single_site_potential = self.load_dca_features_mat(coupling_file, single_site_file, effective_seq_len)
        # =========================================== Bottom@MSA =================================================== 
        

        # =========================================== Top@DSSP ======================================================
        dssp_features = self._load_or_compute_dssp(uniprot_id, pdb_file, sequence)
        dssp_features = dssp_features[:effective_seq_len] # Truncate
        # =========================================== Bottom@DSSP =================================================== 


        # =========================================== Top@atom_coords ======================================================
        coord_result = self._extract_coords_from_pdb(pdb_file)
        if coord_result is None:
            logging.error(f"Could not process PDB for {uniprot_id}, skipping sample.")
            return None # Dataloader will filter this out
        atom_coords, pdb_seq = coord_result
        
        # Ensure sequence from PDB matches sequence from CSV
        if pdb_seq != sequence:
            logging.error(f"Sequence mismatch for {uniprot_id} between PDB and CSV. Skipping sample.")
            return None
        
        # Truncate coordinates to max length
        atom_coords = atom_coords[:effective_seq_len]
        # =========================================== Bottom@atom_coords =================================================== 


        # =========================================== Top@graphec ======================================================
        X_ca = atom_coords[:, 1]
        edge_index = torch_geometric.nn.radius_graph(X_ca, r=self.graphec_radius, loop=True, max_num_neighbors=1000)
        # graphec_node_feature, graphec_edge_feature = self._get_geo_feat(atom_coords, edge_index)
        # =========================================== Bottom@graphec =================================================== 


        active_site_type_label = self.calculate_active_site_types(
            site_label=site_label,
            site_type=site_type,
            sequence_length=effective_seq_len,
        )


        item = {"protein_graph": protein, 
                "reaction_graph": rxn_fclass, 
                "targets": active_site_type_label,
                "msa_tokens": msa_tokens,
                "conservation_scores": conservation_score,
                "coupling_matrix": coupling_matrix,
                "single_site_potentials": single_site_potential,
                "uniprot_id": uniprot_id,
                "canonicalize_rxn_smiles": rxn_smiles,
                "atom_coords": atom_coords,
                "dssp_features": dssp_features,
                # "graphec_node_feature": graphec_node_feature,
                # "graphec_edge_feature": graphec_edge_feature,
                "graphec_edge_index": edge_index
                }
        if self.transform:
            item = self.transform(item)
        assert item["protein_graph"].num_residue.item() == len(item["targets"]), f"Residue count mismatch for {uniprot_id}: graph has {item['protein_graph'].num_residue.item()}, target has {len(item['targets'])}"
        return item



    def save_processed(self):
        
        logging.info(f"Starting to preprocess {len(self)} items in parallel... Saved items will be stored in: {self.preprocessing_dir}")
        num_parallel_workers = self.nb_workers
        indices = range(len(self.dataset_df))

        with ProcessPoolExecutor(max_workers=num_parallel_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_and_save_worker, idx): idx for idx in indices
            }
            logging.info(f"Dispatched {len(future_to_idx)} jobs to {num_parallel_workers} workers.")
            progress_bar = top_tqdm(as_completed(future_to_idx), total=len(indices), desc="Preprocessing and Saving Samples")
            successful_count = 0
            failed_count = 0
            for future in progress_bar:
                idx = future_to_idx[future]
                try:
                    # The result is the final outcome after potential retries.
                    success, message = future.result()
                    if success:
                        successful_count += 1
                    else:
                        failed_count += 1
                        uniprot_id_for_error = self.dataset_df.iloc[idx]['Uniprot ID']
                        # --- MODIFICATION START: Adjust final failure log format ---
                        # Log with the requested format "Failed Processed Sample[label str]: uniprot_id"
                        logging.error(f"Failed Processed Sample[Final Failure]: {uniprot_id_for_error}")
                        # Optionally, log the detailed reason on a separate line for clarity
                        logging.error(f"  - Reason: {message}")
                        # --- MODIFICATION END ---

                except Exception as e:
                    # This catches executor-level errors (like BrokenProcessPool)
                    failed_count += 1
                    uniprot_id_for_error = self.dataset_df.iloc[idx]['Uniprot ID']
                    # --- MODIFICATION START: Adjust final failure log format for executor errors ---
                    logging.error(f"Failed Processed Sample[Executor Error]: {uniprot_id_for_error}")
                    logging.error(f"  - Reason: An executor-level error occurred for item at index {idx}: {e}", exc_info=True)
                    # --- MODIFICATION END ---
                
                progress_bar.set_description(f"Preprocessing (Success: {successful_count}, Failed: {failed_count})")

        logging.info(f"Preprocessing finished. Successful: {successful_count}, Failed: {failed_count}.")
        self.preprocessing = 'load'



    def _process_and_save_worker(self, idx: int) -> Tuple[bool, Optional[str]]:

        max_retries = 3  # Total number of attempts for this single sample
        retry_delay = 0.5  # Seconds to wait between retries
        last_exception = None

        for attempt in range(max_retries):
            try:
                # Step 1: Process the item to get the data dictionary.
                # This reuses the existing getitem logic.
                item = self.getitem(idx)

                # Check if processing was successful but returned an empty object.
                if not item:
                    uniprot_id = self.dataset_df.iloc[idx]['Uniprot ID']
                    error_msg = f"Processing returned None for Uniprot ID: {uniprot_id}"
                    # Raise an exception to be caught by the retry logic
                    raise RuntimeError(error_msg)

                # Step 2: Save the processed item directly from the worker process.
                uniprot_id = item["uniprot_id"]
                rxn_smiles = item["canonicalize_rxn_smiles"]
                rxn_hash = _calculate_rxn_hash(rxn_smiles)
                filename = f"{uniprot_id}_{rxn_hash}.pkl"
                save_path = os.path.join(self.preprocessing_dir, filename)
                torch.save(item, save_path)
                
                # Step 3: If all steps are successful, return True and exit the retry loop.
                return (True, None)

            except Exception as e:
                # If any exception occurs during processing or saving, capture it.
                last_exception = e
                uniprot_id_for_error = self.dataset_df.iloc[idx]['Uniprot ID']
                
                # If this is not the last attempt, log a warning and wait before retrying.
                if attempt < max_retries - 1:
                    logging.warning(
                        f"Worker for item {idx} (Uniprot ID: {uniprot_id_for_error}) failed on attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {retry_delay}s... Error: {e}"
                    )
                    time.sleep(retry_delay)
                # If this was the last attempt, the loop will terminate, and we'll handle the final failure below.

        # Step 4: If the loop completes without a successful return, it means all retries failed.
        uniprot_id_for_error = self.dataset_df.iloc[idx]['Uniprot ID']
        final_error_message = (
            f"Uniprot ID {uniprot_id_for_error}: Failed after {max_retries} attempts. "
            f"Final error: {str(last_exception)}"
        )
        
        # This message will be passed back to the main process.
        return (False, final_error_message)



    def _load_and_filter_data(self) -> pd.DataFrame:
        """
        Now The method behavior is controlled by `self.preprocessing`:
            - 'load': Loads a pre-filtered dataframe.
            - 'save': Performs filtering and saves the result.
            - None: Performs filtering without saving.
        
        
        Loads data from CSV and performs a multi-stage validation process:
        1. Validates protein structure sequences against sequences from the CSV.
        2. Validates and canonicalizes reaction SMILES.
        3. Checks for the existence of all required MSA and DCA files.
        4. Validates the presence(not None) of required columns in the CSV.
        Only samples that pass all checks are retained.

        Returns:
            pd.DataFrame: Filtered dataframe with valid samples only. Valid columns include:
                'Uniprot ID'
                'canonicalize_rxn_smiles'  # 建议使用canonicalize_rxn_smiles键而不是rxn_smiles_single键，来访问某个样本的化学反应
                'Sequence'
                'parsed_site_labels'
                'parsed_site_types'
                'pdb_files'
        """

        # Mode 'load': Directly load the pre-filtered and validated dataframe
        if self.preprocessing == 'load':
            logging.info(f"Loading pre-filtered dataframe from {self.dataframe_check_cache_file}...")
            if not os.path.exists(self.dataframe_check_cache_file):
                raise FileNotFoundError(
                    f"Filtered dataframe cache not found at {self.dataframe_check_cache_file}. "
                    f"Please run the dataset with preprocessing='save' first to create it."
                )
            final_df = pd.read_pickle(self.dataframe_check_cache_file)
            logging.info(f"Loaded {len(final_df)} valid samples from cached dataframe.")
            return final_df

        # Modes 'save' and None: Perform the full validation and filtering process
        logging.info("Starting full dataframe validation and filtering process...")
        # ================== Load the raw dataframe from the CSV file. ======================
        df = pd.read_csv(self.csv_file)
        required_cols = ['Uniprot ID', 'rxn_smiles_single', 'Sequence', 'ec', 'site_labels', 'site_types']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        logging.info(f"Loaded {len(df)} raw entries from {self.csv_file}")


        # ================== (site_labels, site_types) --> (parsed_site_labels, parsed_site_types)========================================
        # Parse site_labels - handle nested list format
        def parse_site_labels(labels_str):
            if pd.isna(labels_str) or labels_str == '':
                return []
            if isinstance(labels_str, str):
                # Parse string representation of nested list like "[[17, 23], [42]]"
                try:
                    return eval(labels_str)  # Use eval to parse nested list
                except Exception as e:
                    logging.warning(f"Failed to parse site_labels: {labels_str}, error: {e}")
                    return []
            elif isinstance(labels_str, (list, tuple)):
                return list(labels_str)
            else:
                return []
        
        # Parse site_types
        def parse_site_types(types_str):
            if pd.isna(types_str) or types_str == '':
                return []
            if isinstance(types_str, str):
                try:
                    return eval(types_str)  # Use eval to parse list
                except Exception as e:
                    logging.warning(f"Failed to parse site_types: {types_str}, error: {e}")
                    return []
            elif isinstance(types_str, (list, tuple)):
                return list(types_str)
            else:
                return []
        
        df['parsed_site_labels'] = df['site_labels'].apply(parse_site_labels)
        df['parsed_site_types'] = df['site_types'].apply(parse_site_types)


        # ================== Uniprot ID -->  pdb_files ========================================
        df['pdb_files'] = df['Uniprot ID'].apply(lambda x: os.path.join(
            self.structure_path, f'AF-{x}-F1-model_v4.pdb'))


        # ================== Structure Sequence Validation ========================================
        logging.info('Checking sequence from structure files...')
        # Create a unique set of PDB files to check, avoiding redundant processing.
        unique_pdb_df = df[['pdb_files']].drop_duplicates().reset_index(drop=True)
        # Use the provided multiprocess checker to get sequences from PDB files.
        unique_pdb_df = multiprocess_structure_check(unique_pdb_df, nb_workers=self.nb_workers)
        # Merge the calculated sequences back into the main dataframe.
        df = pd.merge(df, unique_pdb_df, on='pdb_files', how='left')
        
        # Perform the validation check against the 'Sequence' column.
        # Fill NaN for calculated sequences to avoid errors, an invalid sequence is an empty string.
        df['Sequence_calculated'] = df['Sequence_calculated'].fillna('')
        df['is_alphafolddb_valid'] = (df['Sequence_calculated'] == df['Sequence'])
        valid_count = df['is_alphafolddb_valid'].sum()
        logging.info(f"\nStructure sequence valid: {valid_count}/{len(df)} ({100 * valid_count / len(df):.2f}%)\n")
        

        # ================== Reaction SMILES Validation ========================================
        logging.info("Checking and canonicalizing reaction smiles...")
        # To reuse the checking function, we work with a unique set of reactions.
        # The column `rxn_smiles_single` corresponds to `rxn_smiles` in the original EasIFA logic.
        unique_rxn_df = df[['rxn_smiles_single']].drop_duplicates().rename(
            columns={'rxn_smiles_single': 'rxn_smiles'}
        ).reset_index(drop=True)
        
        # Use the provided multiprocess checker to canonicalize SMILES.
        unique_rxn_df = multiprocess_reaction_check(unique_rxn_df, nb_workers=self.nb_workers)
        
        # Merge the canonicalized SMILES back.
        df = pd.merge(df, unique_rxn_df, left_on='rxn_smiles_single', right_on='rxn_smiles', how='left')
        
        # A reaction is valid if its canonicalized form is not an empty string.
        df['is_reaction_valid'] = df['canonicalize_rxn_smiles'].fillna('') != ''
        valid_count = df['is_reaction_valid'].sum()
        logging.info(f"\nReaction SMILES valid: {valid_count}/{len(df)} ({100 * valid_count / len(df):.2f}%)\n")


        # ================== MSA and DCA File Existence Check ========================================
        logging.info("Checking for MSA and DCA file existence...")
        # Vectorized check for file existence is much faster than iterrows.
        df['msa_exists'] = df['Uniprot ID'].apply(
            lambda uid: os.path.exists(os.path.join(self.msa_dir, f"{uid}.a3m"))
        )
        df['dca_exists'] = df['Uniprot ID'].apply(
            lambda uid: os.path.exists(os.path.join(self.dca_dir, f"{uid}.mat")) and \
                        os.path.exists(os.path.join(self.dca_dir, f"{uid}.single"))
        )
        msa_count = df['msa_exists'].sum()
        dca_count = df['dca_exists'].sum()
        logging.info(f"Found {msa_count}/{len(df)} MSA files and {dca_count}/{len(df)} complete DCA file pairs.")


        # ================== Final Filtering ========================================
        # Combine all validity checks into a single mask.
        final_mask = (
            df['is_alphafolddb_valid'] &
            df['is_reaction_valid'] &
            df['msa_exists'] &
            df['dca_exists'] &
            df['Sequence'].notna() & 
            (df['Sequence'].str.len() > 0) &
            df['Uniprot ID'].notna() &
            df['rxn_smiles_single'].notna() &
            (df['parsed_site_labels'].apply(len) > 0) 

        )
        
        final_df = df[final_mask].reset_index(drop=True)
        logging.info(f"Final dataset size after all checks: {len(final_df)} samples.")

        if self.preprocessing == 'save':
            final_df.to_pickle(self.dataframe_check_cache_file)
            logging.info(f"Saving filtered dataframe to {self.dataframe_check_cache_file}...")
        return final_df
    


    def _setup_protein_transforms(self):
        transforms_func = transforms.transform.Compose([
            transforms.ProteinView(view='residue', keys="protein_graph"),
            transforms.TruncateProtein(
                max_length=self.protein_max_length, random=False, keys='protein_graph')
        ])
        self.transform = transforms_func



    def __len__(self):
        return len(self.dataset_df)



    def _extract_coords_from_pdb(self, pdb_path: str):
        """
        Parses a PDB file to extract atomic coordinates for N, CA, C, O, and the sidechain centroid

        * This function enforces that every standard amino acid residue in the chain must be fully parsable. If any residue is missing backbone atoms or is non-standard, the entire protein is rejected.
        * The function assumes the PDB file contains a single model and a single chain

        Args:
            pdb_path (str): The file path to the PDB file.

        Returns:
            tuple[torch.Tensor, str] or None: 
                - On success: A tuple containing (
                    1. A tensor of shape [num_residues, 5, 3] with coordinates.
                    2. The one-letter amino acid sequence as a string.
                )
                - On failure: None.
        """
        backbone_atoms = ["N", "CA", "C", "O"]
        warnings.filterwarnings("ignore", category=UserWarning)

        parser = PDBParser()
        try:
            structure = parser.get_structure("protein", pdb_path)
        except Exception as e:
            logging.error(f"CRITICAL ERROR: Failed to parse PDB file {pdb_path}. Reason: {e}")
            return None
        
        try:
            # We assume a single model and single chain, as is standard for AlphaFold predictions
            chain = next(structure[0].get_chains())
        except StopIteration:
            logging.error(f"CRITICAL ERROR: No chains found in {pdb_path}.")
            return None
        
        valid_residues = [res for res in chain if is_aa(res.get_resname(), standard=True)]
        expected_num_residues = len(valid_residues)
        if expected_num_residues == 0:
            logging.error(f"CRITICAL ERROR: No standard amino acid residues found in the chain for {pdb_path}.")
            return None
        
        try:
            expected_sequence = "".join([three_to_one(res.get_resname()) for res in valid_residues])
        except KeyError as e:
            logging.error(f"CRITICAL ERROR: Could not convert a three-letter code to one-letter in {pdb_path}. Residue: {e}")
            return None


        all_residue_coords = []
        for residue in valid_residues:
            residue_coords = {atom.get_name(): atom.get_coord() for atom in residue if atom.get_name() in backbone_atoms}

            # Ensure all four backbone atoms were found ---
            if len(residue_coords) != 4:
                missing = set(backbone_atoms) - set(residue_coords.keys())
                logging.error(f"CRITICAL ERROR: In {pdb_path}, residue {residue.get_resname()}{residue.get_id()[1]} is missing backbone atoms: {missing}. Protein rejected.")
                return None
                
            # Calculate sidechain centroid (R)
            sidechain_atoms = [atom for atom in residue if atom.get_name() not in backbone_atoms]
            if sidechain_atoms:
                centroid = np.mean([atom.get_coord() for atom in sidechain_atoms], axis=0)
            else: # Handles Glycine
                centroid = residue_coords["CA"]
            residue_coords["R"] = centroid

            # Assemble coordinates in the specified order
            ordered_coords = np.stack([
                residue_coords["N"],
                residue_coords["CA"],
                residue_coords["C"],
                residue_coords["O"],
                residue_coords["R"]
            ])
            

            all_residue_coords.append(ordered_coords)

        # Final Sanity Check
        if len(all_residue_coords) != expected_num_residues:
            logging.error(f"CRITICAL ERROR: Mismatch in processed ({len(all_residue_coords)}) vs expected ({expected_num_residues}) residues for {pdb_path}. Protein rejected.")
            return None

        final_tensor = torch.from_numpy(np.stack(all_residue_coords)).float()
        
        return final_tensor, expected_sequence



    # =========================================== Top@reaction ====================================================== 
    def _calculate_rxn_features(self, rxn):
        try:
            react, prod = rxn.split(">>")

            react_features_tuple = self._calculate_features(react)
            prod_features_tuple = self._calculate_features(prod)

            return react_features_tuple, prod_features_tuple
        except:
            return None



    def _calculate_features(self, smi):

        # function: rdkit.Chem.MolFromSmiles. This function parses the input SMILES string into [class: rdkit.Chem.rdchem.Mol] object, which internally stores the molecular graph (atoms as nodes and bonds as edges).
        mol = Chem.MolFromSmiles(smi)
        fgraph = self.mol_to_graph(
            mol,
            node_featurizer=self.node_featurizer,
            edge_featurizer=self.edge_featurizer,
            canonical_atom_order=False,
        )
        dgraph = get_adm(mol)

        return fgraph, dgraph
    # =========================================== Bottom@reaction =================================================== 



    # =========================================== Top@MSA ====================================================== 
    def parse_msa_file(self, msa_path: str, max_seqs: int = None, keep_gaps: bool = True, 
                    keep_insertions: bool = False, to_upper: bool = False) -> List[Tuple[str, str]]:
        """
        Parse MSA file.

        确保加载的MSA文件内部对齐
        
        Args:
            msa_path: Path to MSA file (a3m format)
            max_seqs: Maximum number of sequences to return
            keep_gaps: Whether to keep gap characters ('-')
            keep_insertions: Whether to keep lowercase letters (insertions in a3m)
            to_upper: Whether to convert sequences to uppercase
        
        Returns:
            a list of sequences from the MSA file, be like:
                [
                    (label1, sequence1),  # 序列1 - 通常是查询序列
                    (label2, sequence2),  # 序列2
                    (label3, sequence3),  # 序列3
                    ...
                ]
        """
        if not msa_path or not os.path.exists(msa_path):
            return []
            
        try:
            f = open(msa_path, 'r')
                
            msa_seqs = []
            seq = ""
            header = ""
            
            # Parse MSA file line by line
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if seq:
                        msa_seqs.append((header, seq))
                        if max_seqs and len(msa_seqs) >= max_seqs:
                            break
                    header = line[1:]
                    seq = ""
                else:
                    # Process sequence based on parameters
                    if not keep_insertions:
                        # More efficient than regex for simple character removal
                        deletekeys = dict.fromkeys(string.ascii_lowercase)
                        line = line.translate(str.maketrans('', '', ''.join(deletekeys)))
                    if not keep_gaps:
                        line = line.replace('-', '')
                    if to_upper:
                        line = line.upper()
                    seq += line
            
            
            if seq:         # Add the last sequence
                msa_seqs.append((header, seq))

            f.close()

            if msa_seqs:            # Consistency check for sequence lengths (MSA should be aligned)
                seq_lengths = set(len(seq) for _, seq in msa_seqs)
                if len(seq_lengths) > 1:
                    logging.warning(f"MSA sequences have different lengths: {seq_lengths}")
            
            return msa_seqs[:max_seqs]      # Return (label, seq) pairs, limited by max_seqs
            
        except Exception as e:
            logging.error(f"Error parsing MSA file {msa_path}: {e}")
            return []
        


    def parse_and_tokenize_msa(self, msa_path: str, effective_seq_len: int, max_tokens_seqs: int = None) -> Tuple[List[Tuple[str, str]], torch.Tensor]:
        """
        处理 raw MSA 文件
            * 调用self.msa_batch_converter，将raw MSA前 max_tokens_seqs个序列转成msa_tokens
            * 调用self.conservation, 处理raw MSA所有序列转成保守性得分
        
        Args:
            msa_path: MSA file, processed by parse_msa_file, a list of sequences from the MSA file, be like:
                [
                    (label1, sequence1),  # 序列1 - 通常是查询序列
                    (label2, sequence2),  # 序列2
                    (label3, sequence3),  # 序列3
                    ...
                ]
            max_tokens_seqs: Maximum number of sequences to include in tokenized output
            
        Returns:
            Tuple containing:
                - conservation_scores: 1D torch.FloatTensor, [valid_len]
                - msa_tokens: Tensor of tokenized MSA sequences [<=max_tokens_seqs, seq_len_i]
        """
        all_msa_seqs = self.parse_msa_file(msa_path, max_seqs=None, keep_gaps=True)
        
        # 对于tokenization，仅使用指定数量的序列
        if max_tokens_seqs is not None and len(all_msa_seqs) > max_tokens_seqs:
            tokenization_seqs = all_msa_seqs[:max_tokens_seqs]
        else:
            tokenization_seqs = all_msa_seqs
        
        # 使用MSA Transformer的批处理转换器进行tokenization
        '''
        msa_tokens: [batch_size, <=max_tokens_seqs, seq_len_i]
            * 这里是处理一个样本的MSA，所以batch_size==1
            * <=max_seqs表示一个样本的MSA深度可能小于max_seqs，而执行下面的self.msa_batch_converter并没有对单样本MSA深度进行填充
            * seq_len_i的含义是：一个样本对应MSA的第一条序列（即该样本序列）经过截断、映射、填充self.alphabet.prepend_bos和self.alphabet.append_eos得到的
                长度，seq_len_i是一个样本对应的MSA文件内部对齐长度，在一个batch中，可能样本之间的MSA对齐长度不一样，所以用seq_len_i表示。
            * 先对齐单个样本MSA的长度seq_len_i，在批包装时，再对齐一个batch中所有样本的MSA长度seq_len
        '''
        _, _, msa_tokens = self.msa_batch_converter([tokenization_seqs])
        msa_tokens = msa_tokens.squeeze(0)

        # Calculate conservation scores on the full-length, full-depth MSA.
        conservation_scores = self.conservation(all_msa_seqs)
        
        if conservation_scores.size(0) > effective_seq_len:
            conservation_scores = conservation_scores[:effective_seq_len]

        return conservation_scores, msa_tokens



    def load_dca_features_mat(self, coupling_path: str, single_site_path: str, effective_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load DCA features from .mat and .single files.
        
        Args:
            coupling_path: Path to coupling matrix file (.mat)
            single_site_path: Path to single-site potentials file (.single)
            effective_seq_len (int): The target length to which the sequences should be truncated.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - coupling_matrix (torch.Tensor): Truncated to [effective_seq_len, effective_seq_len].
                - single_site_potentials (torch.Tensor): Truncated to [effective_seq_len, 20].
        """
        max_retries = 3
        retry_delay = 0.5  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                # Load coupling matrix from .mat file
                with open(coupling_path, 'r') as f:
                    lines = []
                    for line in f:
                        # Strip whitespace and trailing tabs, then split by tabs
                        cleaned_line = line.strip().rstrip('\t')
                        if cleaned_line:  # Skip empty lines
                            values = [float(x) for x in cleaned_line.split('\t') if x.strip()]
                            lines.append(values)
                # shape: [len(sequence), valid_len]
                coupling_matrix = torch.tensor(lines, dtype=torch.float)
                
                # Load single-site potentials from .single file
                with open(single_site_path, 'r') as f:
                    lines = []
                    for line in f:
                        # Strip whitespace and trailing tabs, then split by tabs
                        cleaned_line = line.strip().rstrip('\t')
                        if cleaned_line:  # Skip empty lines
                            values = [float(x) for x in cleaned_line.split('\t') if x.strip()]
                            if len(values) == 20:  # Should have exactly 20 amino acid values
                                lines.append(values)
                            else:
                                logging.warning(f"Line with {len(values)} values instead of 20 in {single_site_path}")
                # shape: [len(sequence), 20]
                single_site_potentials = torch.tensor(lines, dtype=torch.float)

                if coupling_matrix.size(0) > effective_seq_len:
                    coupling_matrix = coupling_matrix[:effective_seq_len, :effective_seq_len]
                if single_site_potentials.size(0) > effective_seq_len:
                    single_site_potentials = single_site_potentials[:effective_seq_len, :]
                
                # If everything is successful, exit the loop and return
                return coupling_matrix, single_site_potentials
            
            except ValueError as e:
                # Specifically catch ValueError, which is the error from float() conversion
                if attempt < max_retries:
                    logging.warning(f"Attempt {attempt + 1} failed for {os.path.basename(coupling_path)} with error: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    # If it's the last attempt, log the final error and re-raise
                    logging.error(f"All attempts failed for {os.path.basename(coupling_path)}. Final error: {e}")
                    raise
            except Exception as e:
                # Catch other exceptions like FileNotFoundError without retrying
                logging.error(f"Error loading DCA features from .mat/.single files: {e}")
                raise



    def conservation(self, msa_seqs: List[Tuple[str, str]], gap_threshold: float = 0.5, normalize: bool = True) -> torch.FloatTensor:

        """
        为每个残基计算保守性得分（标量值）。

        Compute conservation scores using Henikoff-style weights + Jensen-Shannon Divergence (Capra07).
            – Frequency estimate: vectorized Henikoff weights (O(N·L))
            – Divergence: JSD between weighted freq p and background q
            – Gap handling: positions with gap_fraction >= threshold set to mean−std
            – Normalization: optional Z-score

        * 借鉴于https://github.com/TheApacheCats/al2co

        * 关于从al2co各种方法组合中选出适合预测活性位点任务的论述：https://chatgpt.com/s/t_686fb747545881918938ba9f6172f584, https://chatgpt.com/s/t_686fc15a718481919954959a4eacda04
        
        Args:
            msa_seqs: parse_msa_file函数返回的(label, sequence)元组列表
            gap_threshold: 空位比例阈值，高于此阈值的位置将特殊处理
            normalize: 是否执行Z-score标准化
            
        Returns:
            形状为[L]的保守性得分张量，L为比对长度
        """
        # extract sequences and basic dims
        sequences = [seq for _, seq in msa_seqs]
        if not sequences:
            return torch.tensor([])
        N = len(sequences)
        L = len(sequences[0])

        # amino acids and background freq (SwissProt 2025_03)
        AA = "ACDEFGHIKLMNPQRSTVWY"
        BG_FREQ = torch.tensor([
            0.0825,0.0138,0.0546,0.0671,0.0386,
            0.0707,0.0227,0.0590,0.0579,0.0964,
            0.0241,0.0406,0.0474,0.0393,0.0552,
            0.0665,0.0536,0.0685,0.0110,0.0292
        ], dtype=torch.float32, device='cpu')

        char_to_idx_map = np.full(256, -1, dtype=np.int64)
        for i, char in enumerate(AA):
            char_to_idx_map[ord(char)] = i
            char_to_idx_map[ord(char.lower())] = i

        msa_array = np.array([list(seq) for seq in sequences], dtype='c').view(np.uint8)
        msa_tokens_np = char_to_idx_map[msa_array]
        msa_tokens = torch.from_numpy(msa_tokens_np)

        # valid mask and gap fractions
        valid = (msa_tokens >= 0)
        gap_frac = (~valid).float().mean(dim=0)  # [L]

        # one-hot [N, L, 20]
        one_hot = F.one_hot(msa_tokens.clamp(min=0), num_classes=20).float()
        one_hot = one_hot * valid.unsqueeze(-1).float()  # mask gaps out

        # compute per-column amino acid counts and distinct types
        counts = one_hot.sum(dim=0)                # [L,20]
        types = (counts > 0).sum(dim=1)            # [L]

        # vectorized Henikoff weights:
        #  for each (seq,pos): weight = valid / (count[aa,pos] * types[pos])
        #  then normalize per column
        # gather count_sp: [N,L] = counts[pos, aa]  
        #  note: transpose counts to [20,L] to index by msa_tokens
        counts_T = counts.transpose(0, 1)          # [20, L]
        pos_idx = torch.arange(L)
        # index counts and types
        count_sp = counts_T[msa_tokens.clamp(min=0), pos_idx.unsqueeze(0)]  # [N, L]
        types_p = types.unsqueeze(0).expand(N, -1).float()                  # [N, L]

        # base weights: 1 / (count * types), zero where invalid or gap
        base = count_sp * types_p
        base = base.clamp(min=1e-12)
        weights = valid.float() / base

        # normalize per column so sum_i weights[i,pos] == 1
        col_sum = weights.sum(dim=0, keepdim=True)        # [1, L]
        weights = weights / col_sum

        # compute weighted frequencies p [L,20]
        weighted_one_hot = one_hot * weights.unsqueeze(-1)  # [N, L, 20]
        token_counts = weighted_one_hot.sum(dim=0)          # [L, 20]
        seq_counts = weights.sum(dim=0).unsqueeze(-1)       # [L, 1]
        seq_counts = seq_counts.clamp(min=1e-12)
        p = token_counts / seq_counts                       # [L,20]

        # background q
        q = BG_FREQ.unsqueeze(0).expand(L, -1)              # [L,20]

        # Jensen-Shannon Divergence
        m = 0.5 * (p + q)
        eps = 1e-12
        kl_pm = (p * (p.div(m).clamp(min=eps).log())).sum(dim=1)
        kl_qm = (q * (q.div(m).clamp(min=eps).log())).sum(dim=1)
        jsd = 0.5 * (kl_pm + kl_qm)                         # [L]

        # gap handling: set gappy columns to mean−std of valid
        mask_ok = gap_frac < gap_threshold
        if mask_ok.sum() > 0:
            valid_scores = jsd[mask_ok]
            mu, sigma = valid_scores.mean(), valid_scores.std()
            jsd[~mask_ok] = mu - sigma
            # optional Z-score normalization
            if normalize and sigma > 0:
                jsd = (jsd - mu) / sigma

        return jsd
    # =========================================== Bottom@MSA =================================================== 



    # # =========================================== Top@GraphEC node and edge feature ====================================================== 
    # def _get_geo_feat(self, X: torch.Tensor, edge_index: torch.Tensor):
    #     """
    #     get geometric node features and edge features
    #     """
    #     pos_embeddings = self._positional_embeddings(edge_index)     # [E, 16]
    #     node_angles = self._get_angle(X)
    #     node_dist, edge_dist = self._get_distance(X, edge_index)
    #     node_direction, edge_direction, edge_orientation = self._get_direction_orientation(X, edge_index)

    #     geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    #     geo_edge_feat = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)

    #     return geo_node_feat, geo_edge_feat



    # def _positional_embeddings(self, edge_index, num_embeddings=16):
    #     """
    #     get the positional embeddings
    #     """
    #     d = edge_index[0] - edge_index[1]

    #     frequency = torch.exp(
    #         torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
    #         * -(np.log(10000.0) / num_embeddings)
    #     )
    #     angles = d.unsqueeze(-1) * frequency
    #     PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    #     return PE



    # def _get_angle(self, X, eps=1e-7):
    #     """
    #     get the angle features
    #     """
    #     # psi, omega, phi
    #     X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    #     dX = X[1:] - X[:-1]
    #     U = F.normalize(dX, dim=-1)
    #     u_2 = U[:-2]
    #     u_1 = U[1:-1]
    #     u_0 = U[2:]

    #     # Backbone normals
    #     n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    #     n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    #     # Angle between normals
    #     cosD = torch.sum(n_2 * n_1, -1)
    #     cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    #     D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    #     D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
    #     D = torch.reshape(D, [-1, 3])
    #     dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    #     # alpha, beta, gamma
    #     cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    #     cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    #     D = torch.acos(cosD)
    #     D = F.pad(D, [1, 2])
    #     D = torch.reshape(D, [-1, 3])
    #     bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    #     node_angles = torch.cat((dihedral, bond_angles), 1)
    #     return node_angles # dim = 12



    # def _rbf(self, D, D_min=0., D_max=20., D_count=16):
    #     '''
    #     Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    #     That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    #     '''
    #     D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    #     D_mu = D_mu.view([1, -1])
    #     D_sigma = (D_max - D_min) / D_count
    #     D_expand = torch.unsqueeze(D, -1)

    #     RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    #     return RBF



    # def _get_distance(self, X, edge_index):
    #     """
    #     get the distance features
    #     """
    #     atom_N = X[:,0]  # [L, 3]
    #     atom_Ca = X[:,1]
    #     atom_C = X[:,2]
    #     atom_O = X[:,3]
    #     atom_R = X[:,4]

    #     node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
    #     node_dist = []
    #     for pair in node_list:
    #         atom1, atom2 = pair.split('-')
    #         E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
    #         rbf = self._rbf(E_vectors.norm(dim=-1))
    #         node_dist.append(rbf)
    #     node_dist = torch.cat(node_dist, dim=-1) # dim = [N, 10 * 16]

    #     atom_list = ["N", "Ca", "C", "O", "R"]
    #     edge_dist = []
    #     for atom1 in atom_list:
    #         for atom2 in atom_list:
    #             E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
    #             rbf = self._rbf(E_vectors.norm(dim=-1))
    #             edge_dist.append(rbf)
    #     edge_dist = torch.cat(edge_dist, dim=-1) # dim = [E, 25 * 16]

    #     return node_dist, edge_dist



    # def _get_direction_orientation(self, X, edge_index): # N, CA, C, O, R
    #     """
    #     get the direction features
    #     """
    #     X_N = X[:,0]  # [L, 3]
    #     X_Ca = X[:,1]
    #     X_C = X[:,2]
    #     u = F.normalize(X_Ca - X_N, dim=-1)
    #     v = F.normalize(X_C - X_Ca, dim=-1)
    #     b = F.normalize(u - v, dim=-1)
    #     n = F.normalize(torch.cross(u, v), dim=-1)
    #     local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)

    #     node_j, node_i = edge_index

    #     t = F.normalize(X[:, [0,2,3,4]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    #     node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1) # [L, 4 * 3]

    #     t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    #     edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    #     t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    #     edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3]
    #     edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    #     r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
    #     edge_orientation = self._quaternions(r) # [E, 4]

    #     return node_direction, edge_direction, edge_orientation



    # def _quaternions(self, R):
    #     """ Convert a batch of 3D rotations [R] to quaternions [Q]
    #         R [N,3,3]
    #         Q [N,4]
    #     """
    #     diag = torch.diagonal(R, dim1=-2, dim2=-1)
    #     Rxx, Ryy, Rzz = diag.unbind(-1)
    #     magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
    #         Rxx - Ryy - Rzz,
    #         - Rxx + Ryy - Rzz,
    #         - Rxx - Ryy + Rzz
    #     ], -1)))
    #     _R = lambda i,j: R[:,i,j]
    #     signs = torch.sign(torch.stack([
    #         _R(2,1) - _R(1,2),
    #         _R(0,2) - _R(2,0),
    #         _R(1,0) - _R(0,1)
    #     ], -1))
    #     xyz = signs * magnitudes
    #     # The relu enforces a non-negative trace
    #     w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    #     Q = torch.cat((xyz, w), -1)
    #     Q = F.normalize(Q, dim=-1)

    #     return Q
    # # =========================================== Bottom@GraphEC node and edge feature =================================================== 



    # =========================================== Top@DSSP ======================================================
    def _load_or_compute_dssp(self, uniprot_id: str, pdb_file: str, ref_seq: str):
        tensor_path = os.path.join(self.dssp_dir, f"{uniprot_id}.tensor")
        if os.path.exists(tensor_path):
            return torch.load(tensor_path)

        dssp_out_path = os.path.join(self.dssp_dir, f"{uniprot_id}.dssp")
        mkdssp_executable = os.path.join(self.dssp_exec_path, "mkdssp") # Assuming mkdssp is in the directory
        command = f"{mkdssp_executable} -i {pdb_file} -o {dssp_out_path}"
        exit_code = os.system(command)
        if exit_code != 0 or not os.path.exists(dssp_out_path):
            logging.error(f"mkdssp failed for {uniprot_id} with exit code {exit_code}. PDB: {pdb_file}")
            # Return a zero tensor as a fallback to avoid crashing the whole process
            return torch.zeros(len(ref_seq), 9, dtype=torch.float32)

        dssp_seq, dssp_matrix = self._process_dssp(dssp_out_path)
        
        if dssp_seq != ref_seq:
            logging.warning(f"DSSP sequence mismatch for {uniprot_id}. Aligning...")
            dssp_matrix = self._match_dssp(dssp_seq, dssp_matrix, ref_seq)
        
        if len(dssp_matrix) != len(ref_seq):
             logging.error(f"DSSP alignment failed for {uniprot_id}. Expected len {len(ref_seq)}, got {len(dssp_matrix)}.")
             return torch.zeros(len(ref_seq), 9, dtype=torch.float32)

        dssp_tensor = torch.tensor(np.array(dssp_matrix), dtype=torch.float32)
        torch.save(dssp_tensor, tensor_path)
        return dssp_tensor
    


    def _process_dssp(self, dssp_file: str):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"
        SS_type = "HBEGITSC"
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

        with open(dssp_file, "r") as f:
            lines = f.readlines()

        seq = ""
        dssp_feature = []

        p = 0
        while lines[p].strip()[0] != "#":
            p += 1
        for i in range(p + 1, len(lines)):
            aa = lines[i][13]
            if aa == "!" or aa == "*":
                continue
            seq += aa
            SS = lines[i][16]
            if SS == " ":
                SS = "C"
            SS_vec = np.zeros(8)
            SS_vec[SS_type.find(SS)] = 1
            ACC = float(lines[i][34:38].strip())
            ASA = min(1, ACC / rASA_std[aa_type.find(aa)])
            dssp_feature.append(np.concatenate((np.array([ASA]), SS_vec)))

        return seq, dssp_feature



    def _match_dssp(self, seq: str, dssp: list, ref_seq: str):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        padded_item = np.zeros(9)

        new_dssp = []
        for aa in seq:
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0))

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-":
                continue
            matched_dssp.append(new_dssp[i])

        return matched_dssp
    # =========================================== Bottom@DSSP =================================================== 



    # =========================================== Top@PDB ======================================================
    def process_target(self, pdb_path, parse_hetatom=False, center=True):
        """
        读取一个PDB文件，并将其处理成模型所需的标准特征格式。
        """
        # Read target pdb and extract features.
        target_struct = self.parse_pdb(pdb_path, parse_hetatom=parse_hetatom)

        # Zero-center positions
        ca_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True)
        if not center:
            ca_center = 0
        xyz_27 = torch.from_numpy(target_struct["xyz"] - ca_center)
        seq_orig = torch.from_numpy(target_struct["seq"])
        mask_27 = torch.from_numpy(target_struct["mask"])
        
        out = {
            "xyz_27": xyz_27,
            "mask_27": mask_27,
            "seq": seq_orig,
            "pdb_idx": target_struct["pdb_idx"],
        }
        if parse_hetatom:
            out["xyz_het"] = target_struct["xyz_het"]
            out["info_het"] = target_struct["info_het"]
        return out
    


    def parse_pdb(self, filename, **kwargs):
        """extract xyz coords for all heavy atoms"""
        with open(filename,"r") as f:
            lines=f.readlines()
        return self.parse_pdb_lines(lines, **kwargs)
    


    def parse_pdb_lines(self, lines, parse_hetatom=False, ignore_het_h=True):
        """
        Returns: 
            out 是一个字典：
                "xyz":
                    内容: 蛋白质所有可能的27个原子的坐标。
                    结构: NumPy 数组 (np.ndarray)。
                    形状: (L, 27, 3)
                "mask":
                    内容: 一个布尔掩码，标记 xyz 数组中哪些原子是真实存在于PDB文件中的。
                    结构: NumPy 布尔数组 (np.ndarray)。
                    形状: (L, 27)。mask[i, j] 为 True 表示第 i 个残基的第 j 个原子坐标是真实的。
                "idx":
                    内容: PDB文件中每个残基的原始编号。
                    结构: NumPy 整数数组 (np.ndarray)。
                    形状: (L,)。例如 [10, 11, 12, ...]。
                "seq":
                    内容: 蛋白质的氨基酸序列，已经从三字母码转换成了数字表示。
                    结构: NumPy 整数数组 (np.ndarray)。
                    形状: (L,)。
                "pdb_idx":
                    内容: 每个残基的唯一标识符，由链ID和残基编号组成。
                    结构: Python 列表 (list)，每个元素是一个元组 (chain_id, residue_number)。
                    形状: 长度为 L 的列表，例如 [('A', 10), ('A', 11), ...]。
                "xyz_het" (可选):
                    内容: 如果 parse_hetatom=True，此键存在，包含所有杂原子的三维坐标。
                    结构: NumPy 数组 (np.ndarray)。
                    形状: (num_het_atoms, 3)。
                "info_het" (可选):
                    内容: 如果 parse_hetatom=True，此键存在，包含每个杂原子的详细信息。
                    结构: Python 列表 (list)，每个元素是一个字典，如 {'idx': ..., 'atom_id': ..., ...}。
                    形状: 长度为 num_het_atoms 的列表。
        """
        # indices of residues observed in the structure
        res, pdb_idx = [],[]
        for l in lines:
            if l[:4] == "ATOM" and l[12:16].strip() == "CA":
                res.append((l[22:26], l[17:20]))
                # chain letter, res num
                pdb_idx.append((l[21:22].strip(), int(l[22:26].strip())))
        seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]
        pdb_idx = [
            (l[21:22].strip(), int(l[22:26].strip()))
            for l in lines
            if l[:4] == "ATOM" and l[12:16].strip() == "CA"
        ]  # chain letter, res num

        # up to 27 atoms per residue
        # 每个残基最多27个原子
        xyz = np.full((len(res), 27, 3), np.nan, dtype=np.float32)
        for l in lines:
            if l[:4] != "ATOM":
                continue
            chain, resNo, atom, aa = (
                l[21:22],
                int(l[22:26]),
                " " + l[12:16].strip().ljust(3),
                l[17:20],
            )
            if (chain,resNo) in pdb_idx:
                idx = pdb_idx.index((chain, resNo))
                for i_atm, tgtatm in enumerate(
                    aa2long[aa2num[aa]]
                    ):
                    if (
                        tgtatm is not None and tgtatm.strip() == atom.strip()
                        ):  # ignore whitespace
                        xyz[idx, i_atm, :] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                        break

        # save atom mask
        mask = np.logical_not(np.isnan(xyz[..., 0]))
        xyz[np.isnan(xyz[..., 0])] = 0.0

        # remove duplicated (chain, resi)
        new_idx = []
        i_unique = []
        for i, idx in enumerate(pdb_idx):
            if idx not in new_idx:
                new_idx.append(idx)
                i_unique.append(i)

        pdb_idx = new_idx
        xyz = xyz[i_unique]
        mask = mask[i_unique]
        seq = np.array(seq)[i_unique]

        out = {
            "xyz": xyz,  # cartesian coordinates, [Lx27,3]
            "mask": mask,  # mask showing which atoms are present in the PDB file, [Lx27]
            "idx": np.array(
                [i[1] for i in pdb_idx]
            ),  # residue numbers in the PDB file, [L]
            "seq": np.array(seq),  # amino acid sequence, [L]
            "pdb_idx": pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
        }

        # heteroatoms (ligands, etc)
        if parse_hetatom:
            xyz_het, info_het = [], []
            for l in lines:
                if l[:6] == "HETATM" and not (ignore_het_h and l[77] == "H"):
                    info_het.append(
                        dict(
                            idx=int(l[7:11]),
                            atom_id=l[12:16],
                            atom_type=l[77],
                            name=l[16:20],
                        )
                    )
                    xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            out["xyz_het"] = np.array(xyz_het)
            out["info_het"] = info_het

        return out
    # =========================================== Bottom@PDB ===================================================


    def calculate_active_site_types(self, site_label, site_type, sequence_length):
        """
        Create active site labels tensor from site labels and types.
        Note: Site label starts from 1，比如[[74], [128]]中，表示的是第74个残基和第128个残基是活性位点，而不是第75个和第129个。
        
        Args:
            site_labels: List of site label lists, e.g., [[17, 23], [42], [74]]
            site_types: List of site types corresponding to each site label, e.g., [1, 1, 2]
            sequence_length: effective Length of the sequence, not the original sequence length.

            
        Returns:
            active_site_type_labels：Tensor
                * 其长度effective_seq_len = min(len(sequence), self.protein_max_length)
                * 0表示非活性位点，非0表示活性位点，更进一步1表示结合位点，2表示催化位点，3表示其他位点
                * active_site_type_labels中不包含special token(cls/bos, eos, pad)
        """
        assert len(site_label) == len(site_type)

        active_site_type_labels = torch.zeros((sequence_length,))
        for one_site, tps in zip(site_label, site_type): 
            if len(one_site) == 1:
                # Single position (e.g., [74])
                idx = one_site[0] - 1  # Convert to 0-indexed
                if 0 <= idx < sequence_length:
                    active_site_type_labels[idx] = tps
            elif len(one_site) == 2:
                # Range of positions (e.g., [17, 23])
                start, end = one_site[0] - 1, one_site[1] - 1  # Convert to 0-indexed
                for idx in range(start, end + 1):
                    if 0 <= idx < sequence_length:
                        active_site_type_labels[idx] = tps
            else:
                raise ValueError("The label of active site is not standard !!!")
        return active_site_type_labels



    def get_esm_alphabet(self, model_name: str):
        """
        Create an ESM alphabet based on model name without loading the model weights.
        
        Args:
            model_name: Name of the ESM model (e.g., "esm2_t33_650M_UR50D") or Name of the ESM model
            
        Returns:
            Alphabet: The appropriate alphabet for the specified model
        """
        # Handle ESM2 models
        if model_name.startswith("esm2_"):
            # All ESM2 models use the ESM-1b alphabet
            return Alphabet.from_architecture("ESM-1b")
        
        # Handle MSA Transformer models
        elif "msa" in model_name.lower():
            return Alphabet.from_architecture("MSA Transformer")
        
        # Handle ESM-1b models
        elif model_name.startswith("esm1b_") or "esm1b" in model_name:
            return Alphabet.from_architecture("ESM-1b")
        
        # Default to ESM-1b alphabet (most commonly used)
        else:
            return Alphabet.from_architecture("ESM-1b")
    


class EnzymeActiveSiteDataset3(EnzymeMSAStructureDataset):
    """
    * Design for ClusterSiteTrainer/ClusterSite
    * Expected dataset format:
        * CSV file with columns: 'Uniprot ID', 'rxn_smiles_single', 'Sequence', 'ec', 'site_labels', 'site_types'.
            * 'Uniprot ID' be like L0E4H0
            * 'rxn_smiles_single' be like Cc1ccccc1.Cc1ccccc1>>Cc1ccccc(C(=O)O)c1, not include enzyme amino acid sequence between the reactants and products.
            * 'Sequence' be like MSRQLSRARPATVLGAMEMGRRMDAPTSAAVTRAFLERGHTEIDTAFVYSEGQSETILGGLGL
            * 'site_labels' be like [[17, 23], [42], [74], [172], [168]], where each sublist represents a range or single residue index of active sites. Site label starts from 1，比如[[74], [128]]中，表示的是第74个残基和第128个残基是活性位点，而不是第75个和第129个。
            * 'site_types' be like [1, 1, 1, 1, 2], where 1 indicates binding site and 2 indicates catalytic site and 3 indicates other site, 0 indicates non-active site.
            * 'ec' be like 1.1.1.104.
        * 每个样本的'ec'列只包含一个EC number，如果一个酶有多种EC number，请在数据预处理阶段拆分成多个样本，每个样本对应一个EC number。
    """
    def __init__(
        self,
        csv_file: str,
        structure_path: str,
        msa_dir: str,
        dca_dir: str,
        dssp_exec_path: str,
        dssp_dir: str,
        graphec_radius: float = 10.0,
        protein_max_length: int = 600,
        max_msa_sequences: int = 128,
        split: str = "train",
        num_samples: Optional[int] = None,
        num_workers: int = 16,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        msa_model_name: str = "esm_msa1b_t12_100M_UR50S",
        preprocessing: Optional[str] = None,
        preprocessing_dir: Optional[str] = None,
        filter_incomplete_ec: bool = True,
        **kwargs,
    ):
        """
        Args:
            filter_incomplete_ec (bool): If True, filter out EC numbers containing '-' (e.g., 1.1.1.-). 
        """
        self.filter_incomplete_ec = filter_incomplete_ec
        self.ec_str_to_int = {}
        self.idx_to_ec_int = {}
        super().__init__(
            csv_file=csv_file,
            structure_path=structure_path,
            msa_dir=msa_dir,
            dca_dir=dca_dir,
            dssp_exec_path=dssp_exec_path,
            dssp_dir=dssp_dir,
            graphec_radius=graphec_radius,
            protein_max_length=protein_max_length,
            max_msa_sequences=max_msa_sequences,
            split=split,
            num_samples=num_samples,
            num_workers=num_workers,
            esm_model_name=esm_model_name,
            msa_model_name=msa_model_name,
            preprocessing=preprocessing,
            preprocessing_dir=preprocessing_dir,
            **kwargs,
        )
        self._build_ec_mapping()


    def _load_and_filter_data(self) -> pd.DataFrame:
        """
        *_load_and_filter_data函数返回的DataFrame是后续所有操作的数据基础。
        """
        logging.info("Starting _load_and_filter_data for EnzymeActiveSiteDataset3...")
        
        df = pd.read_csv(self.csv_file)
        required_cols = ['Uniprot ID', 'rxn_smiles_single', 'Sequence', 'ec', 'site_labels', 'site_types']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        # ================== (site_labels, site_types) --> (parsed_site_labels, parsed_site_types)========================================
        # Parse site_labels - handle nested list format
        def parse_site_labels(labels_str):
            if pd.isna(labels_str) or labels_str == '':
                return []
            if isinstance(labels_str, str):
                # Parse string representation of nested list like "[[17, 23], [42]]"
                try:
                    return eval(labels_str)  # Use eval to parse nested list
                except Exception as e:
                    logging.warning(f"Failed to parse site_labels: {labels_str}, error: {e}")
                    return []
            elif isinstance(labels_str, (list, tuple)):
                return list(labels_str)
            else:
                return []
        
        # Parse site_types
        def parse_site_types(types_str):
            if pd.isna(types_str) or types_str == '':
                return []
            if isinstance(types_str, str):
                try:
                    return eval(types_str)  # Use eval to parse list
                except Exception as e:
                    logging.warning(f"Failed to parse site_types: {types_str}, error: {e}")
                    return []
            elif isinstance(types_str, (list, tuple)):
                return list(types_str)
            else:
                return []
        
        df['parsed_site_labels'] = df['site_labels'].apply(parse_site_labels)
        df['parsed_site_types'] = df['site_types'].apply(parse_site_types)

        # ================== Uniprot ID -->  pdb_files ========================================
        def get_pdb_path(uid):
            v6_path = os.path.join(self.structure_path, f'AF-{uid}-F1-model_v6.pdb')
            if os.path.exists(v6_path):
                return v6_path
            return os.path.join(self.structure_path, f'AF-{uid}-F1-model_v4.pdb')
        
        df['pdb_files'] = df['Uniprot ID'].apply(get_pdb_path)


        # ================== Structure Sequence Validation ========================================
        logging.info('Checking sequence from structure files...')
        # Create a unique set of PDB files to check, avoiding redundant processing.
        unique_pdb_df = df[['pdb_files']].drop_duplicates().reset_index(drop=True)
        # Use the provided multiprocess checker to get sequences from PDB files.
        unique_pdb_df = multiprocess_structure_check(unique_pdb_df, nb_workers=self.nb_workers)
        # Merge the calculated sequences back into the main dataframe.
        df = pd.merge(df, unique_pdb_df, on='pdb_files', how='left')
        
        # Perform the validation check against the 'Sequence' column.
        # Fill NaN for calculated sequences to avoid errors, an invalid sequence is an empty string.
        df['Sequence_calculated'] = df['Sequence_calculated'].fillna('')
        df['is_alphafolddb_valid'] = (df['Sequence_calculated'] == df['Sequence'])
        valid_count = df['is_alphafolddb_valid'].sum()
        logging.info(f"\nStructure sequence valid: {valid_count}/{len(df)} ({100 * valid_count / len(df):.2f}%)\n")


        # ================== Reaction SMILES Validation ========================================
        logging.info("Checking and canonicalizing reaction smiles...")
        # To reuse the checking function, we work with a unique set of reactions.
        unique_rxn_df = df[['rxn_smiles_single']].drop_duplicates().rename(
            columns={'rxn_smiles_single': 'rxn_smiles'}
        ).reset_index(drop=True)
        
        # Use the provided multiprocess checker to canonicalize SMILES.
        unique_rxn_df = multiprocess_reaction_check(unique_rxn_df, nb_workers=self.nb_workers)
        
        # Merge the canonicalized SMILES back.
        df = pd.merge(df, unique_rxn_df, left_on='rxn_smiles_single', right_on='rxn_smiles', how='left')
        
        # A reaction is valid if its canonicalized form is not an empty string.
        df['is_reaction_valid'] = df['canonicalize_rxn_smiles'].fillna('') != ''
        valid_rxn_count = df['is_reaction_valid'].sum()
        logging.info(f"\nReaction SMILES valid: {valid_rxn_count}/{len(df)} ({100 * valid_rxn_count / len(df):.2f}%)\n")


        # ================== Generate Labels Vector and Length ========================================
        df['Sequence_length'] = df['Sequence'].apply(lambda x: min(len(str(x)), self.protein_max_length))
        
        if self.nb_workers > 0:
            pandarallel.initialize(nb_workers=self.nb_workers, progress_bar=False, verbose=0)
            df['labels'] = df.parallel_apply(self.parse_site_labels, axis=1)
        else:
            top_tqdm.pandas(desc="Generating labels")
            df['labels'] = df.progress_apply(self.parse_site_labels, axis=1)

        # ================== EC ========================================
        df['valid_ec_list'] = df['ec'].apply(lambda x: self.parse_ec(x, filter_incomplete=self.filter_incomplete_ec))

        # ================== Final Filtering ========================================
        final_mask = (
            df['Sequence'].notna() & 
            (df['Sequence'].str.len() > 0) &
            df['Uniprot ID'].notna() &
            df['is_alphafolddb_valid'] &
            df['is_reaction_valid'] &
            (df['parsed_site_labels'].apply(len) > 0) &
            df['labels'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0 and x.max() > 0) &        # 过滤掉标签生成失败（全0）的脏数据
            (df['valid_ec_list'].apply(len) > 0)    # 确保至少有一个有效的完整 EC 号
        )
        
        final_df = df[final_mask].reset_index(drop=True)
        logging.info(f"Final dataset size: {len(final_df)} samples.")
        
        return final_df



    def getitem(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.dataset_df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.dataset_df)}")
        row = self.dataset_df.iloc[idx]
        uniprot_id = row['Uniprot ID']
        sequence = row['Sequence']
        pdb_file = row['pdb_files']
        labels = row['labels']
        rxn_smiles = row['canonicalize_rxn_smiles']
        effective_seq_len = min(len(sequence), self.protein_max_length)

        # =========================================== Top@Sequence + Structure ====================================================== 
        protein = data.Protein.from_pdb(pdb_file, self.kwargs)
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        # =========================================== Bottom@Sequence + Structure =================================================== 

        # =========================================== Top@Reaction ====================================================== 
        reaction_features = self._calculate_rxn_features(rxn_smiles)
        rxn_fclass = ReactionFeatures(reaction_features, rxn_smiles)
        # =========================================== Bottom@Reaction =================================================== 

        item = {
            "protein_graph": protein,
            "reaction_graph": rxn_fclass,
            "targets": torch.from_numpy(labels).long(),
            "uniprot_id": uniprot_id,
            "canonicalize_rxn_smiles": rxn_smiles,
            "ec_int": self.idx_to_ec_int.get(idx, 0)
        }

        if self.transform:
            item = self.transform(item)
        assert item["protein_graph"].num_residue.item() == len(item["targets"]), f"Residue count mismatch for {uniprot_id}: graph has {item['protein_graph'].num_residue.item()}, target has {len(item['targets'])}"
        return item


    @staticmethod
    def parse_site_labels(row) -> np.ndarray:
        """
        根据 row['parsed_site_labels'] / row['parsed_site_types'] / row['Sequence_length']构造一个 0-based 的 labels 向量:labels[i] ∈ {0,1,2,3}
        若解析失败或字段缺失，则返回全 0。全0labels被视为脏样本，应该被过滤掉。
        """
        seq_len = int(row['Sequence_length'])
        labels = np.zeros(seq_len, dtype=np.int64)

        site_indices = row['parsed_site_labels']
        site_types = row['parsed_site_types']

        if not (isinstance(site_indices, list) and isinstance(site_types, list)):
            return labels
        if len(site_indices) != len(site_types):
            return labels

        for i, indices_list in enumerate(site_indices):
            site_type = int(site_types[i])
            if len(indices_list) == 1:
                res_idx = int(indices_list[0])
                if 1 <= res_idx <= seq_len:
                    labels[res_idx - 1] = site_type  # 1-based -> 0-based
            elif len(indices_list) == 2:
                start_idx, end_idx = int(indices_list[0]), int(indices_list[1])
                for res_idx in range(start_idx, end_idx + 1):
                    if 1 <= res_idx <= seq_len:
                        labels[res_idx - 1] = site_type
        return labels
    

    @staticmethod
    def parse_ec(ec_number: str, filter_incomplete: bool = True) -> List[str]:
        """
        将 ec_number 解析为 EC 列表，并过滤掉含 '-' 的残缺 EC。
        例如:
            "1.1.1.1;2.3.4.5" -> ["1.1.1.1", "2.3.4.5"]
            "1.1.1.1,2.3.4.5" -> ["1.1.1.1", "2.3.4.5"]
            "1.1.1.-"         -> []  (因含 '-' 被过滤)
        """
        if pd.isna(ec_number):
            return []
        ec_str = str(ec_number).strip()
        if not ec_str:
            return []

        if ';' in ec_str:
            parts = ec_str.split(';')
        elif ',' in ec_str:
            parts = ec_str.split(',')
        else:
            parts = [ec_str]

        ec_list = [p.strip() for p in parts if p.strip()]
        if filter_incomplete:
            valid_ec_list = [ec for ec in ec_list if '-' not in ec]
        else:
            valid_ec_list = ec_list
        return valid_ec_list



    def _build_ec_mapping(self):
        """
        Builds a mapping from EC strings to integers and assigns an EC integer to each sample.
        Required for SiteSupConTrainer to perform hard positive/negative mining based on EC identity.
        """
        logging.info("Building EC mapping indices...")
        unique_ecs = set()
        
        # 1. Collect all unique ECs
        for _, row in self.dataset_df.iterrows():
            ecs = row.get('valid_ec_list', [])
            if ecs and len(ecs) > 0:
                unique_ecs.add(ecs[0])
        
        # 2. Create string-to-int mapping (1-based, 0 reserved for unknown/background)
        sorted_ecs = sorted(list(unique_ecs))
        self.ec_str_to_int = {ec: i + 1 for i, ec in enumerate(sorted_ecs)}
        
        # 3. Assign int ID to each sample index
        valid_count = 0
        for idx, row in self.dataset_df.iterrows():
            ecs = row.get('valid_ec_list', [])
            if ecs and len(ecs) > 0:
                # Use dataset dataframe index which matches __getitem__ idx due to reset_index
                self.idx_to_ec_int[idx] = self.ec_str_to_int[ecs[0]]
                valid_count += 1
            else:
                self.idx_to_ec_int[idx] = 0 # 0 for unknown or no EC
        
        logging.info(f"Mapped {valid_count}/{len(self.dataset_df)} samples to {len(sorted_ecs)} unique EC classes.")

        # 4. Build cluster_to_indices and valid_clusters for GroupedBatchSampler
        self.cluster_to_indices = defaultdict(list)
        for idx, ec_int in self.idx_to_ec_int.items():
            if ec_int > 0:  # Only include valid ECs (exclude 0 for unknown)
                self.cluster_to_indices[ec_int].append(idx)
        
        # Only include clusters with at least one sample
        self.valid_clusters = [c for c, indices in self.cluster_to_indices.items() if len(indices) > 0]
        logging.info(f"Created {len(self.valid_clusters)} valid clusters for GroupedBatchSampler.")

        

class EnzymeActiveSiteBinaryDataset3(EnzymeActiveSiteDataset3):
    """
    Binary version of EnzymeActiveSiteDataset3.
    Maps all active site types (1, 2, 3, etc.) to 1. 0 remains 0 (non-active).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def parse_site_labels(row) -> np.ndarray:
        """
        Binary version: labels[i] ∈ {0, 1}
        """
        seq_len = int(row['Sequence_length'])
        labels = np.zeros(seq_len, dtype=np.int64)

        site_indices = row['parsed_site_labels']
        site_types = row['parsed_site_types']

        if not (isinstance(site_indices, list) and isinstance(site_types, list)):
            return labels
        if len(site_indices) != len(site_types):
            return labels

        for i, indices_list in enumerate(site_indices):
            # In binary mode, we ignore the specific site_type value (as long as it is > 0)
            # and assign label 1.
            # However, the original logic checks site_type. 
            # We assume inputs in CSV are valid active sites.
            
            if len(indices_list) == 1:
                res_idx = int(indices_list[0])
                if 1 <= res_idx <= seq_len:
                    labels[res_idx - 1] = 1  # Always 1 for active sites
            elif len(indices_list) == 2:
                start_idx, end_idx = int(indices_list[0]), int(indices_list[1])
                for res_idx in range(start_idx, end_idx + 1):
                    if 1 <= res_idx <= seq_len:
                        labels[res_idx - 1] = 1 # Always 1 for active sites
        return labels
    


def collate_rxn_features(batch):
    """
    output:
        batch_data: a dict 
            key rts_bg: Batched reactant molecular graph containing merged nodes and edges of all reactants. [class: dgl.heterograph.DGLGraph] object
            key rts_adms: torch.Tensor, [batch_size, max_atoms, max_atoms]. Reactant atomic distance matrices, padded to the size of the largest molecule in the batch
            key rts_node_feats: torch.Tensor, [total_nodes, node_feat_dim]. Atomic features of all reactants, where total_nodes is the sum of all atoms in the batch
            key rts_edge_feats: torch.Tensor, [total_edges, edge_feat_dim]. Bond features of all reactants, where total_edges is the sum of all bonds in the batch

            key rxn_smiles: A list where each element is the reaction smiles corresponding to a reaction sample in a batch.
    """
    react_fgraphs = [x.react_fgraph for x in batch]
    react_dgraphs = [x.react_dgraph for x in batch]
    prod_fgraphs = [x.prod_fgraph for x in batch]
    prod_dgraphs = [x.prod_dgraph for x in batch]
    rxn_values = [str(x.rxn) for x in batch]

    # [function: dgl.batch] returns [class: dgl.heterograph.DGLGraph] object.
    rts_bg = dgl.batch(react_fgraphs)
    pds_bg = dgl.batch(prod_fgraphs)
    rts_bg.set_n_initializer(dgl.init.zero_initializer)
    pds_bg.set_n_initializer(dgl.init.zero_initializer)

    rts_adms = pad_atom_distance_matrix(react_dgraphs)
    pds_adms = pad_atom_distance_matrix(prod_dgraphs)

    # ndata and edata are attributes of the [class: dgl.heterograph.DGLGraph] object, and their type is dictionary (dict)
    rts_node_feats, rts_edge_feats = rts_bg.ndata.pop("h"), rts_bg.edata.pop("e")
    pds_node_feats, pds_edge_feats = pds_bg.ndata.pop("h"), pds_bg.edata.pop("e")

    return {
        "rts_bg": rts_bg,
        "rts_adms": rts_adms,
        "rts_node_feats": rts_node_feats,
        "rts_edge_feats": rts_edge_feats,
        "pds_bg": pds_bg,
        "pds_adms": pds_adms,
        "pds_node_feats": pds_node_feats,
        "pds_edge_feats": pds_edge_feats,
        "rxn_smiles": rxn_values,
    }



def enzyme_rxn_collate(batch):
    """
    Convert any list of same nested container into a container of tensors.

    For instances of :class:`data.Graph <torchdrug.data.Graph>`, they are collated
    by :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.

    Parameters:
        batch (list): list of samples with the same nested container
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        shapes = [x.shape for x in batch]
        return torch.cat(batch, 0, out=out), torch.tensor(shapes)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    
    # class: torchdrug.data.graph.Graph
    # torchdrug.data.graph.Graph is the parent class of torchdrug.data.protein.Protein
    elif isinstance(elem, data.Graph):

        # function: torchdrug.data.graph.Graph.pack. This function returns [class: torchdrug.data.graph.PackedGraph] object.
        return elem.pack(batch)
    
    # class: collections.abc.Mapping
    # collections.abc.Mapping is the parent class of dict
    elif isinstance(elem, Mapping):
        return {key: enzyme_rxn_collate([d[key] for d in batch]) for key in elem}
    
    # class: collections.abc.Sequence
    # collections.abc.Sequence is the parent class of list, tuple, etc.
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("Each element in list of batch should be of equal size")
        return [enzyme_rxn_collate(samples) for samples in zip(*batch)]

    elif isinstance(elem, ReactionFeatures):
        return collate_rxn_features(batch)

    raise TypeError("Can't collate data with type `%s`" % type(elem))



def enzyme_rxn_collate_extract(batch, handle_msa=False):
    """
    input:
        batch: some dicts handled by the get_item function of the dataset class
    output:
        batch_data: a dict 
            key protein_graph: [class: torchdrug.data.graph.PackedGraph] object
            key reaction_graph: A dict, is the output of the collate_rxn_features function
            key targets: A list where each element is a binary tensor of the active site position corresponding to a protein sample
            key protein_len: A tensor where each element is the length of the amino acid residue sequence of each protein in a batch
            key pdb_file (optional): A list of strings, each element corresponds to the PDB file path of a sample.
            key uniprot_id (optional): A list of strings, each element corresponds to the UniProt ID of a sample.
            key msa (optional): A list, each element is the MSA of one sample, be like:
                [
                    [ (label1, sequence1), (label2, sequence2), ... ],   # 第一个样本的MSA
                    [ (label1, sequence1), (label2, sequence2), ... ],   # 第二个样本的MSA
                    ...
                ]

    """
    original_batch_size = len(batch)
    batch = [x for x in batch if x]
    filtered_count = original_batch_size - len(batch)

    if filtered_count > 0:
        print(f"Filtered out {filtered_count} failed samples from batch of {original_batch_size}")
    
    # If entire batch is empty, return None to signal the training loop to skip this batch
    if not batch:
        print("Warning: Entire batch is empty after filtering failed samples, skipping batch")
        return None
    
    msa_data = None
    if handle_msa:
        msa_data = [x.pop("msa") for x in batch if "msa" in x]

    # Extract uniprot_id if present in batch items
    uniprot_ids = None
    if "uniprot_id" in batch[0]:
        uniprot_ids = [x.pop("uniprot_id") for x in batch]

    try:
        batch_data = enzyme_rxn_collate(batch)
        assert isinstance(batch_data, dict)
        
        if handle_msa and msa_data:     # 保持msa_data原始结构
            batch_data["msa"] = msa_data

        # Add uniprot_id to batch_data if available
        if uniprot_ids:
            batch_data["uniprot_id"] = uniprot_ids

        if "targets" in batch_data:
            if isinstance(batch_data["targets"], tuple):
                targets, size = batch_data["targets"]
                batch_data["targets"] = targets
                batch_data["protein_len"] = size.view(-1)
        return batch_data
    except Exception as e:
        print(f"Error in enzyme_rxn_collate: {e}")
        return None



def collate_enzyme_batch(batch: List[Dict[str, Any]]) -> Union[Dict[str, Any], None]:

    # Filter out samples that failed to load (e.g., returned None from __getitem__)
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        print("Warning: Entire batch is empty after filtering failed samples. Skipping.")
        return None

    # =========================================== General item ======================================================
    generic_collate_keys = [
        "protein_graph", 
        "targets", 
        "conservation_scores",
        "atom_coords",
        "dssp_features",
    ]
    
    items_for_generic_collate = []
    for sample in batch:
        item_subset = {key: sample[key] for key in generic_collate_keys if key in sample}
        items_for_generic_collate.append(item_subset)
    
    collated_batch = enzyme_rxn_collate(items_for_generic_collate)

    for key in ["targets", "conservation_scores", "atom_coords", "dssp_features"]:
        if key in collated_batch and isinstance(collated_batch[key], tuple):
            collated_batch[key], _ = collated_batch[key]

    # =========================================== reaction_graph ======================================================
    if 'reaction_graph' in batch[0]:
        reaction_features_list = [sample['reaction_graph'] for sample in batch]
        collated_batch['reaction_graph'] = collate_rxn_features(reaction_features_list)

    # =========================================== MSA Tokens: Pad to a dense 3D tensor. ======================================================
    msa_key = 'msa' if 'msa' in batch[0] else 'msa_tokens'
    if msa_key in batch[0]:
        # Determine max dimensions for padding
        max_msa_seqs = max(sample[msa_key].size(0) for sample in batch if sample[msa_key].nelement() > 0) if any(s[msa_key].nelement() > 0 for s in batch) else 1
        max_msa_len = max(sample[msa_key].size(1) for sample in batch if sample[msa_key].nelement() > 0) if any(s[msa_key].nelement() > 0 for s in batch) else 1
        
        # 根据键选择正确的填充值 (20 for 'msa', 1 for 'msa_tokens')
        padding_idx = 20 if msa_key == 'msa' else 1
        msa_padded = torch.full((len(batch), max_msa_seqs, max_msa_len), padding_idx, dtype=torch.long)
        
        for i, sample in enumerate(batch):
            msa = sample[msa_key]
            if msa.nelement() > 0:
                num_seqs, seq_len = msa.shape
                msa_padded[i, :num_seqs, :seq_len] = msa
        
        collated_batch[msa_key] = msa_padded

    # =========================================== Coupling Matrix & Single Site Potentials: Collect into lists. ======================================================
    if 'coupling_matrix' in batch[0]:
        collated_batch['coupling_matrices'] = [sample['coupling_matrix'] for sample in batch]
    if 'single_site_potentials' in batch[0]:
        collated_batch['single_site_potentials'] = [sample['single_site_potentials'] for sample in batch]

    # =========================================== UniProt ID: Collect into a list of strings. ======================================================
    if 'uniprot_id' in batch[0]:
        collated_batch['uniprot_ids'] = [sample['uniprot_id'] for sample in batch]

    # =========================================== Canonical SMILES ======================================================
    if 'canonicalize_rxn_smiles' in batch[0]:
        collated_batch['canonicalize_rxn_smiles'] = [sample['canonicalize_rxn_smiles'] for sample in batch]
    if 'reaction_embedding' in batch[0]:
        collated_batch['reaction_embedding'] = torch.stack([sample['reaction_embedding'] for sample in batch])


    # =========================================== GraphEC ======================================================
    if 'graphec_edge_index' in batch[0]:
        
        # Get the number of residues for each graph from the already collated `protein_graph`.
        graph = collated_batch['protein_graph']
        num_residues_per_graph = graph.num_residues.tolist()

        # Extract the lists of tensors from the original uncollated batch.
        edge_indices_list = [sample['graphec_edge_index'] for sample in batch]

        # Calculate node offsets for creating the batched graph.
        node_offsets = torch.cumsum(torch.tensor([0] + num_residues_per_graph), 0)[:-1]

        # Create batched edge indices by applying the offsets.
        batched_edge_indices = torch.cat(
            [edge_index + offset for edge_index, offset in zip(edge_indices_list, node_offsets)], 
            dim=1
        )
        # Create the batch_id vector required by the GNN encoder.
        batch_id = torch.repeat_interleave(
            torch.arange(len(num_residues_per_graph)), 
            torch.tensor(num_residues_per_graph)
        )
        collated_batch['batched_graphec_edge_indices'] = batched_edge_indices
        collated_batch['graphec_batch_id'] = batch_id


    # =========================================== seq_mask ======================================================
    batch_size = len(batch)
    seq_lens = collated_batch['protein_graph'].num_residues
    max_len = seq_lens.max().item()
    idx = torch.arange(max_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    seq_mask = (idx < seq_lens.unsqueeze(1))
    collated_batch['seq_mask'] = seq_mask

    # =========================================== msa_mask ======================================================
    if 'msa' in collated_batch:
        max_msa_depth = collated_batch['msa'].shape[1]
        msa_depths = torch.tensor([s['msa'].shape[0] for s in batch if 'msa' in s and s['msa'].nelement() > 0], dtype=torch.long)
        msa_mask = (torch.arange(max_msa_depth).unsqueeze(0).repeat(batch_size, 1) < msa_depths.unsqueeze(1))
        collated_batch['msa_mask'] = msa_mask
    

    # =========================================== x_0 ======================================================
    if 'x_0' in batch[0]:
        xyz_27_padded = torch.zeros((batch_size, max_len, 27, 3), dtype=torch.float32)
        mask_27_padded = torch.zeros((batch_size, max_len, 27), dtype=torch.bool)
        seq_padded = torch.full((batch_size, max_len), 20, dtype=torch.long)
        chain_gap = 200
        processed_pdb_idx_padded = torch.full((batch_size, max_len), 0, dtype=torch.long)

        for i, sample in enumerate(batch):
            x0_data = sample['x_0']
            length = x0_data['seq'].shape[0]

            xyz_27_padded[i, :length] = x0_data['xyz_27']
            mask_27_padded[i, :length] = x0_data['mask_27']
            seq_padded[i, :length] = x0_data['seq']
            
            pdb_indices = sample['x_0'].get('pdb_idx')
            if pdb_indices:
                assert length == len(pdb_indices), f"Sample {i}: seq length ({length}) and pdb_idx length ({len(pdb_indices)}) mismatch."
                chains, res_nums = zip(*pdb_indices)
                chain_ids_np = np.array([ord(c) if c else ord('A') for c in chains], dtype=np.int64)
                chain_indices = torch.from_numpy(chain_ids_np - ord('A'))
                res_nums_tensor = torch.tensor(res_nums, dtype=torch.long)
                sample_processed_idx = res_nums_tensor + chain_indices * chain_gap
                processed_pdb_idx_padded[i, :length] = sample_processed_idx

        collated_batch['xyz_27'] = xyz_27_padded
        collated_batch['mask_27'] = mask_27_padded
        collated_batch['seq'] = seq_padded
        collated_batch['pdb_idx'] = processed_pdb_idx_padded
    

    # =========================================== ec_number ======================================================
    if 'ec_number' in batch[0]:
        collated_batch['ec_number'] = torch.stack([s['ec_number'] for s in batch])


    # =========================================== Cluster ID Expansion ==============================================
    if 'ec_int' in batch[0]:
        protein_ec_ints = []
        residue_ec_ints = []
        
        for i, sample in enumerate(batch):
            c_int = sample['ec_int']

            protein_ec_ints.append(c_int)
            num_residues = sample['protein_graph'].num_residue.item()
            residue_ec_ints.append(torch.full((num_residues,), c_int, dtype=torch.long))
            
        collated_batch['protein_ec_ints'] = torch.tensor(protein_ec_ints, dtype=torch.long)   # Stack protein-level IDs: [B]
        collated_batch['residue_ec_ints'] = torch.cat(residue_ec_ints, dim=0)                 # Concatenate residue-level IDs: [Total_Residues]
    
    return collated_batch


 
class ResidueFeatureDataset(Dataset):
    def __init__(self, initial_residue_dict):
        self.protein_items = [
            (uniprot_id, res_tensor) 
            for uniprot_id, res_tensor in initial_residue_dict.items() 
            if isinstance(res_tensor, torch.Tensor)
        ]

    def __len__(self):
        return len(self.protein_items)

    def __getitem__(self, idx):
        uniprot_id, residue_tensor = self.protein_items[idx]
        return uniprot_id, residue_tensor



def collate_residue_features(batch):
    uniprot_ids = [item[0] for item in batch]
    residue_tensors = [item[1] for item in batch]
    lengths = [tensor.shape[0] for tensor in residue_tensors]

    concatenated_residues = torch.cat(residue_tensors, dim=0)
    return uniprot_ids, concatenated_residues, lengths



class MultiPosNeg_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, n_pos, n_neg, protein_features_dict):
        """
        Args:
            mine_neg: 每个 EC 的“硬负样本 EC 候选池”
                mine_neg = {
                    '1.1.1.1': {
                        'weights': [w1, w2, ..., wk],   # 归一化后的采样权重
                        'negative': ['1.1.1.2', '1.1.1.3', ..., '2.3.4.5']  # 最近的 k 个“非自身”EC
                    },
                    '2.7.1.1': {
                        'weights': [...],
                        'negative': ['2.7.1.2', '3.4.21.4', ...]
                    },
                    # ...
                }
            protein_features_dict: 蛋白质特征字典
                protein_features_dict = {
                    'P19367': tensor([D]),  # 维度 D = ECEmbeddingNet 的输入维度，例如 128
                    'P52789': tensor([D]),
                    ...
                }
        """
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = sorted(list(ec_id.keys()))
        print(f"Dataset initialized with {len(self.full_list)} unique EC numbers.")

        try:
            self.esm_embeddings = protein_features_dict
        except FileNotFoundError:
            raise FileNotFoundError("ESM embedding dictionary file not found. ")

        # Pool
        self.mine_neg = mine_neg

        """
        self.positive_pools：每个锚点蛋白的“正样本蛋白池”,对于每个蛋白 anchor，找出所有“与它共享至少一个 EC 的其他蛋白”，作为正样本候选池。
        self.positive_pools = {
            'P1': ['P2', 'P3'],
            'P2': ['P1'],
            'P3': ['P1'],
            'P4': [],              # 没有其它同 EC 蛋白
        }
        """
        self.positive_pools = {}
        for anchor_id in top_tqdm(self.id_ec.keys(), desc="Building Positive Pools", disable=True):
            candidate_ecs = self.id_ec[anchor_id]
            pool = set()
            for ec in candidate_ecs:
                # 确保 ec 存在于 ec_id 字典中
                if ec in self.ec_id:
                    pool.update(self.ec_id[ec])
            pool.discard(anchor_id) # 移除锚点自身
            self.positive_pools[anchor_id] = list(pool)
        

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        all_ids = self.sample_ids(index)
        try:
            data = [self.esm_embeddings[pid].unsqueeze(0) for pid in all_ids]
        except KeyError as e:
            print(f"Error: Protein ID '{e.args[0]}' not found in the pre-loaded ESM embeddings dictionary.")
            raise e
        return torch.cat(data, dim=0)


    def sample_ids(self, index) -> List[str]:
        """
        采样逻辑：根据索引生成 Anchor, Positive, Negative 的 ID 列表。
        """
        # 1 select Anchor
        anchor_ec = self.full_list[index]
        anchor_id = random.choice(self.ec_id[anchor_ec])

        # 2 sample positive samples
        pos_candidates = self.positive_pools[anchor_id]
        if not pos_candidates:
            pos_ids = [anchor_id] * self.n_pos # $$这里是隐患，后续需要处理
        else:
            pos_ids = random.choices(pos_candidates, k=self.n_pos)

        # 3 sample negative samples
        anchor_ecs = self.id_ec[anchor_id]
        anchor_ecs_count = len(anchor_ecs)
        samples_per_ec = self.n_neg // anchor_ecs_count     # 计算每个 EC 应该采样多少个负样本
        remaining_samples = self.n_neg % anchor_ecs_count
        neg_ids = []
        for i, sampling_ec in enumerate(anchor_ecs):
            if sampling_ec not in self.mine_neg:
                raise ValueError(f"Sampling EC '{sampling_ec}' not found in mine_neg dictionary.")
            
            n_samples = samples_per_ec + (1 if i < remaining_samples else 0)    # 前 remaining_samples 个 EC 多采样 1 个
            if n_samples <= 0:  continue                                        # 这一轮本来就不需要负样本，直接跳过

            neg_ec_pool = self.mine_neg[sampling_ec]['negative']
            weights = self.mine_neg[sampling_ec]['weights']
            
            sampled_ecs = []
            max_trials = 20         # 防止死循环的最大尝试次数
            trials = 0
            while len(sampled_ecs) < n_samples and trials < max_trials:
                num_to_sample = (n_samples - len(sampled_ecs)) * 2
                candidates = random.choices(neg_ec_pool, weights=weights, k=num_to_sample)
                valid_ecs = [ec for ec in candidates if ec not in anchor_ecs]
                if valid_ecs:
                    sampled_ecs.extend(valid_ecs)
                trials += 1

            if len(sampled_ecs) < n_samples:        # 如果多次尝试之后还是不够，就从全局 EC 池中（排除掉 anchor_ecs）补齐，避免死循环
                global_neg_pool = [ec for ec in self.full_list if ec not in anchor_ecs]
                if not global_neg_pool:             # 极端兜底：连全局都没有非 anchor 的 EC，就只能用某个 anchor_ec 占位，保证代码能跑下去
                    fallback_ec = anchor_ecs[0]
                    sampled_ecs.extend([fallback_ec] * (n_samples - len(sampled_ecs)))
                else:
                    sampled_ecs.extend(random.choices(global_neg_pool, k=(n_samples - len(sampled_ecs))))
            neg_ids.extend([random.choice(self.ec_id[ec]) for ec in sampled_ecs[:n_samples]])   # 取前 n_samples 个，并为每个 EC 随机选择一个蛋白质 ID  $$这里是去前 n_samples 个，还是随机挑选 n_samples 个

        # 4 fetch data
        all_ids = [anchor_id] + pos_ids + neg_ids
        return all_ids
    


class MultiPosNeg_dataset_with_mine_SITE(torch.utils.data.Dataset):
    """
    * MultiPosNeg_dataset_with_mine_SITE(torch.utils.data.Dataset)只用于训练
    """
    def __init__(self, n_pos: int, n_neg: int, residue_features_dict: Dict[str, torch.Tensor], dataframe: pd.DataFrame, num_workers: int, hard_neg_ratio: float, cluster_mode: str, ec_level: int):
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.hard_neg_ratio = hard_neg_ratio
        self.residue_embeddings_dict = residue_features_dict
        
        print("Building data pools with logic consistent with ECEmbeddingNetTrainer...")
        self.id_ec, ec_id_set, self.ec_site_pools, _, _, _, _, _ = build_data_pools(
            dataframe=dataframe,
            embedding_dict=self.residue_embeddings_dict,
            num_workers=num_workers,
            ec_level=ec_level,
            cluster_mode=cluster_mode
        )
        self.ec_id = {key: list(value) for key, value in ec_id_set.items()}

        self.anchors = []
        for ec, type_pools in self.ec_site_pools.items():
            for site_type, pointers in type_pools.items():
                if site_type > 0:
                    for i in range(len(pointers)):
                        # 锚点信息: (EC号, 位点类型, 在该池中的索引)
                        # 这个索引将用于在 __getitem__ 中定位指针
                        self.anchors.append((ec, site_type, i))
        print(f"Data processing complete. Found {len(self.ec_id)} unique ECs and {len(self.anchors)} active site anchors.")


        self.hard_neg_pool = {}
        for ec, type_pools in self.ec_site_pools.items():
            # 当前 EC 下所有活性类型（site_type > 0）
            active_types = [t for t in type_pools.keys() if t > 0]
            for anchor_type in active_types:
                hard_list = []
                for t in active_types:
                    if t != anchor_type:
                        hard_list.extend(type_pools[t])
                self.hard_neg_pool[(ec, anchor_type)] = hard_list



    def __len__(self):
        return len(self.anchors)



    def __getitem__(self, index):
        """
        Returns:
            torch.Tensor: A tensor of shape (1 + n_pos + n_neg, embedding_dim) containing the anchor,
                          positive samples, and negative samples.
        """
        anchor_ec, anchor_type, anchor_pool_idx = self.anchors[index]
        anchor_pointer = self.ec_site_pools[anchor_ec][anchor_type][anchor_pool_idx]
        anchor_tensor = self.residue_embeddings_dict[anchor_pointer[0]][anchor_pointer[1]]

        # Positive Sampling
        pos_pool_pointers = self.ec_site_pools[anchor_ec][anchor_type]
        sampled_pos_pointers = random.choices(pos_pool_pointers, k=self.n_pos)
        pos_tensors = [self.residue_embeddings_dict[p[0]][p[1]] for p in sampled_pos_pointers]

        # Negative Sampling
        sampled_neg_pointers = []
        target_hard_neg_count = int(self.n_neg * self.hard_neg_ratio)

        hard_pool = self.hard_neg_pool.get((anchor_ec, anchor_type), [])
        if target_hard_neg_count > 0 and hard_pool:
            num_to_sample_hard = min(target_hard_neg_count, len(hard_pool))
            sampled_neg_pointers.extend(random.sample(hard_pool, k=num_to_sample_hard))

        if (self.n_neg - len(sampled_neg_pointers)) > 0 and self.ec_site_pools[anchor_ec].get(0, []):
            sampled_neg_pointers.extend(random.choices(self.ec_site_pools[anchor_ec].get(0, []), k=(self.n_neg - len(sampled_neg_pointers))))

        if (self.n_neg - len(sampled_neg_pointers)) > 0:
            combined_backup_pool = []
            for t, pool in self.ec_site_pools[anchor_ec].items():
                if t != anchor_type and pool:
                    combined_backup_pool.extend(pool)
            if combined_backup_pool:
                sampled_neg_pointers.extend(random.choices(combined_backup_pool, k=(self.n_neg - len(sampled_neg_pointers))))
        
        neg_tensors = [self.residue_embeddings_dict[p[0]][p[1]] for p in sampled_neg_pointers]


        # Final check to ensure we have enough negative samples
        if (self.n_neg - len(neg_tensors)) > 0:
            print(f"Warning: Filling {(self.n_neg - len(neg_tensors))} negative samples with positives for anchor from EC {anchor_ec}.")
            neg_tensors.extend(random.choices(pos_tensors, k=(self.n_neg - len(neg_tensors))))


        # Combine all tensors
        all_tensors = [anchor_tensor.unsqueeze(0)] + \
                      [t.unsqueeze(0) for t in pos_tensors] + \
                      [t.unsqueeze(0) for t in neg_tensors]
        
        return torch.cat(all_tensors, dim=0)



class GroupedBatchSampler(Sampler):
    """
    通用分组 Batch Sampler。$$当前GroupedBatchSampler的sample方法需要根据实验结果进一步优化
    
    设计目标：
        1. 支持按 Cluster（无论是 EC 还是 Reaction Cluster）进行采样。
        2. 优先保证同一个 Batch 内的样本属于同一个 Cluster。
        3. 解决了原 ECWeightedBatchSampler 中可能存在的漏数据问题（通过 Queue 和 Epoch 遍历机制）。
    
    工作流程：
        1. 在每个 Epoch 开始时，打乱 Cluster 的顺序。
        2. 对于每个 Cluster，将其所有样本索引打乱放入队列。
        3. 依次取出 Cluster，从其队列中提取样本填充 Batch。
        4. 如果当前 Cluster 队列空了，则丢弃该 Cluster，继续处理下一个 Cluster。
        5. 如果当前 Cluster 队列没空但 Batch 已满，则将该 Cluster 移到队尾（Round-Robin），等待下一次轮转。
    """
    def __init__(self, 
                 dataset, 
                 batch_size: int, 
                 seed: int = 42):
        """
        Args:
            dataset: 必须是 EnzymeActiveSiteDataset3 实例，且 cluster_mode 不为 None。需要访问 dataset.valid_clusters 和 dataset.cluster_to_indices。
            batch_size: 每个 Batch 的样本数。
            seed: 随机种子。
        """
        if not hasattr(dataset, 'valid_clusters') or not hasattr(dataset, 'cluster_to_indices'):
            raise ValueError("Dataset must have 'valid_clusters' and 'cluster_to_indices' attributes. "
                             "Ensure dataset is initialized with a valid cluster_mode.")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.clusters = dataset.valid_clusters
        self.cluster_indices = {c: list(idxs) for c, idxs in dataset.cluster_to_indices.items() if c in self.clusters}
        
        # 计算总 Batch 数 (近似值，用于 tqdm 显示)
        total_samples = sum(len(idxs) for idxs in self.cluster_indices.values())
        self.num_batches = total_samples // batch_size
        self.rng = np.random.default_rng(seed)
        
    def __iter__(self):
        cluster_queues = {c: deque(self.rng.permutation(idxs)) for c, idxs in self.cluster_indices.items()}     # 初始化每个 Cluster 的洗牌队列
        active_clusters = list(self.clusters)                       # 激活 Cluster 列表并洗牌 (决定 Cluster 的处理顺序)
        self.rng.shuffle(active_clusters)
        current_batch = []
        
        while active_clusters:                                      # 循环填充 Batch
            c = active_clusters[0]                                  # 取出当前处理的 Cluster
            queue = cluster_queues[c]
            while queue and len(current_batch) < self.batch_size:   # 尽可能多地从当前 Cluster 取样，直到 Batch 满 或 该 Cluster 队列空
                current_batch.append(queue.popleft())
            
            if not queue:
                active_clusters.pop(0)                              # 如果当前 Cluster 的样本已耗尽，从活跃列表中移除
            else:
                # 如果 Cluster 还有剩余样本但 Batch 满了，将该 Cluster 移到队尾
                # 这样可以防止一个超大 Cluster 连续占用太多 Batch，增加 Batch 间的多样性
                active_clusters.append(active_clusters.pop(0))
            
            if len(current_batch) == self.batch_size:               # 如果 Batch 填满了，产出
                yield current_batch
                current_batch = []
        
        # 处理剩余样本 (Drop last or yield)
        # 这里选择 yield，确保所有样本都被处理
        if len(current_batch) > 0:
            yield current_batch

    def __len__(self):
        return self.num_batches



def reaction_clusters(
    input_csv: str,
    structure_path: str,
    output_csv: str,
    k_clusters: int = 2000,
    max_seq_length: int = 600,
    batch_size: int = 96,
    num_workers: int = 64,
    gpu: str = "cuda:0",
    kmeans_model_path: Optional[str] = None,
):
    """
    Build reaction clusters (proxy ECs) based on RXNFP fingerprints.

    Pipeline:
    1) 用 EnzymeActiveSiteDataset3 加载并清洗训练数据（包含 canonicalize_rxn_smiles 和 valid_ec_list）。
    2) 读取 dataset_df['canonicalize_rxn_smiles']。
    3) 用 RXNFP 预训练编码器得到反应指纹（BERT 表示）。
    4) 在指纹空间上运行 KMeans 聚类。
    5) 将簇 ID 写回 CSV，得到 df['rxn_cluster_label']。
    6) （可选）如果指定 kmeans_model_path，则把 KMeans 模型持久化到磁盘，以便测试集复用。
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not os.path.exists(structure_path):
        raise FileNotFoundError(f"Structure path not found: {structure_path}")

    print("Initializing EnzymeActiveSiteDataset3 to load cleaned training data...")
    dataset = EnzymeActiveSiteDataset3(
        csv_file=input_csv,
        structure_path=structure_path,
        msa_dir="",                 # deprecated
        dca_dir="",                 # deprecated
        dssp_exec_path="",          # deprecated
        dssp_dir="",                # deprecated
        graphec_radius=0.0,         # deprecated
        protein_max_length=max_seq_length,
        max_msa_sequences=0,        # deprecated
        split="train",              # deprecated
        num_samples=None,           # deprecated
        esm_model_name="esm2_t33_650M_UR50D",
        msa_model_name="esm_msa1b_t12_100M_UR50S",
        num_workers=num_workers,
        preprocessing=None,         # deprecated
        preprocessing_dir=None,     # deprecated
        filter_incomplete_ec=True,
    )

    df = dataset.dataset_df.copy()
    print(f"Loaded cleaned dataset_df with {len(df)} samples.")

    # Check for SMILES column
    smiles_col = "canonicalize_rxn_smiles"
    if smiles_col not in df.columns:
        raise ValueError(f"Expected column '{smiles_col}' in DataFrame.")

    smiles_series = df[smiles_col].fillna("")
    smiles_series = smiles_series[smiles_series != ""]
    unique_smiles = smiles_series.unique().tolist()
    print(f"Found {len(unique_smiles)} unique canonical reactions.")

    if len(unique_smiles) == 0:
        raise ValueError("No non-empty canonical reaction SMILES found in dataset_df.")

    # 1. Build RXNFP pretrained encoder
    print("Loading RXNFP default pretrained model...")
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_default_model_and_tokenizer()
    model = model.to(device)
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)


    # 2. Generate RXNFP fingerprints for unique reactions
    rxn_emb_dict = {}
    print("Generating RXNFP embeddings for unique reactions...")
    for i in top_tqdm(range(0, len(unique_smiles), batch_size), desc="RXNFP Embedding"):
        batch_smi = unique_smiles[i : i + batch_size]
        embeddings = rxnfp_generator.convert_batch(batch_smi)
        for smi, emb in zip(batch_smi, embeddings):
            rxn_emb_dict[smi] = np.asarray(emb, dtype=np.float32)

    # 3. Prepare matrix for clustering
    X = np.stack([rxn_emb_dict[smi] for smi in unique_smiles], axis=0)
    print(f"Embedding matrix shape: {X.shape}")

    # 4. KMeans in RXNFP space
    print(f"Clustering into {k_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=k_clusters,
        random_state=42,
        n_init=10,
        verbose=0,
    )
    cluster_ids = kmeans.fit_predict(X)
    if kmeans_model_path is not None:
        print(f"Saving trained KMeans model to {kmeans_model_path} ...")
        with open(kmeans_model_path, "wb") as f:
            pickle.dump(kmeans, f)

    # 5. Map SMILES -> cluster label
    smiles_to_label = {
        smi: f"RXN_{cid:04d}" for smi, cid in zip(unique_smiles, cluster_ids)
    }

    # 6. Write labels back
    df["rxn_cluster_label"] = df[smiles_col].map(smiles_to_label)

    before = len(df)
    df = df[df["rxn_cluster_label"].notna()].reset_index(drop=True)
    after = len(df)
    print(f"[cluster] kept {after}/{before} rows after assigning cluster labels.")

    print(f"Saving clustered CSV to {output_csv} ...")
    df.to_csv(output_csv, index=False)
    print("Done.")



def evaluate_reaction_clusters(
    input_csv: str,
    cluster_stats_csv: str = "cluster_ec_stats.csv",
    ec_stats_csv: str = "ec_cluster_stats.csv",
):
    """
    用 4级EC 作为“真值标签”，评估 rxn_cluster_label 聚类效果。
    Args:
        input_csv: 输入 CSV 文件路径，需包含 'valid_ec_list' 和 'rxn_cluster_label' 列。
            - valid_ec_list: 形如 "['1.1.1.1']" 或 "['1.1.1.1', '1.1.1.104']"
            - rxn_cluster_label: 形如 "RXN_0259"

    Returns:
      1) Prints global homogeneity / completeness / v_measure to terminal.
      2) Saves per-cluster and per-EC stats to CSV files.
            - cluster_stats_csv: 输出每个聚类的 EC 纯度统计 CSV 文件路径。
            - ec_stats_csv: 输出每个 EC 的聚类分散度统计 CSV 文件路径。
    """
    def _parse_valid_ec_list(cell):
        """
        将字符串形式的 valid_ec_list 解析为 Python list 并取“主 EC 标签”。

        例：
            "['1.1.1.1']"         -> "1.1.1.1"
            "['1.1.1.1','2.3.4.5']" -> "1.1.1.1"  (简单起见，取第一个)
            "[]"                  -> None
        """
        if pd.isna(cell):
            return None
        if isinstance(cell, list):
            ec_list = cell
        else:
            try:
                ec_list = ast.literal_eval(str(cell))
            except Exception:
                return None

        if not isinstance(ec_list, (list, tuple)) or len(ec_list) == 0:
            return None
        # Simply take the first EC as the "main EC label"
        ec = str(ec_list[0]).strip()
        return ec if ec else None

    def _parse_cluster_label(x):
        """
        将 'RXN_0259' -> 259 的整数 ID，方便 sklearn 使用。
        如果格式异常，则返回 None。
        """
        if pd.isna(x):
            return None
        x = str(x).strip()
        if not x:
            return None
        # Expect format RXN_0001
        if x.startswith("RXN_"):
            try:
                return int(x.split("_")[1])
            except Exception:
                return None
        try:
            return int(x)
        except Exception:
            return None

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows.")

    # ===== 1. Parse main EC label & integer Cluster ID =====
    if "valid_ec_list" not in df.columns:
        raise ValueError("Column 'valid_ec_list' not found in CSV.")
    if "rxn_cluster_label" not in df.columns:
        raise ValueError("Column 'rxn_cluster_label' not found in CSV.")

    df["main_ec"] = df["valid_ec_list"].apply(_parse_valid_ec_list)
    df["cluster_id"] = df["rxn_cluster_label"].apply(_parse_cluster_label)

    before = len(df)
    df = df[df["main_ec"].notna() & df["cluster_id"].notna()].reset_index(drop=True)
    after = len(df)
    print(f"Valid rows for evaluation: {after}/{before}")

    if after == 0:
        raise ValueError("No valid rows after parsing 'valid_ec_list' and 'rxn_cluster_label'.")

    # Map main_ec to discrete integer labels for sklearn
    ec_to_int = {ec: idx for idx, ec in enumerate(sorted(df["main_ec"].unique()))}
    df["ec_id"] = df["main_ec"].map(ec_to_int)

    y_true = df["ec_id"].to_numpy()
    y_pred = df["cluster_id"].to_numpy()

    # ===== 2. Compute Global Metrics =====
    # 这是评估聚类质量的三个核心指标（取值范围均为 0~1，越高越好）：
    # *   **Homogeneity (同质性)**：
    #     *   **含义**：每个聚类是否只包含某**一种** EC 的样本？
    #     *   **解读**：如果得分为 1.0，说明每个聚类都很“纯”，没有混杂不同的酶。
    # *   **Completeness (完整性)**：
    #     *   **含义**：同**一种** EC 的所有样本是否都被分到了**同一个**聚类中？
    #     *   **解读**：如果得分为 1.0，说明同一种酶没有被拆散到不同的聚类里。
    # *   **V-measure**：
    #     *   **含义**：同质性和完整性的调和平均数（类似 F1-score）。
    #     *   **解读**：综合评估聚类效果。
    h = homogeneity_score(y_true, y_pred)
    c = completeness_score(y_true, y_pred)
    v = v_measure_score(y_true, y_pred)

    print("\n=== Global clustering metrics (EC as ground-truth) ===")
    print(f"Homogeneity  : {h:.4f}  (Each cluster contains only members of a single class)")
    print(f"Completeness : {c:.4f}  (All members of a given class are assigned to the same cluster)")
    print(f"V-measure    : {v:.4f}  (Harmonic mean of H & C)\n")

    # ===== 3. Per-Cluster Statistics (Purity) =====
    cluster_ec_stats = []
    for cluster_id, sub_df in df.groupby("cluster_id"):
        size = len(sub_df)
        ec_counts = sub_df["main_ec"].value_counts()
        top_ec = ec_counts.index[0]
        top_count = int(ec_counts.iloc[0])
        purity = top_count / size 

        cluster_ec_stats.append({
            "cluster_id": cluster_id,
            "cluster_size": size,
            "num_unique_ec": len(ec_counts),
            "top_ec": top_ec,
            "top_ec_count": top_count,
            "top_ec_purity": purity,
        })

    cluster_ec_stats_df = pd.DataFrame(cluster_ec_stats).sort_values(
        by=["top_ec_purity", "cluster_size"], ascending=[False, False]
    )
    print(f"Saving per-cluster EC stats to {cluster_stats_csv} ...")
    cluster_ec_stats_df.to_csv(cluster_stats_csv, index=False)

    # ===== 4. Per-EC Statistics (Dispersion) =====
    ec_cluster_stats = []
    for ec, sub_df in df.groupby("main_ec"):
        total = len(sub_df)
        cluster_counts = sub_df["cluster_id"].value_counts()
        
        max_cluster_id = int(cluster_counts.index[0])
        max_cluster_count = int(cluster_counts.iloc[0])
        max_cluster_ratio = max_cluster_count / total

        ec_cluster_stats.append({
            "ec": ec,
            "num_samples": total,
            "num_clusters": len(cluster_counts),
            "max_cluster_id": max_cluster_id,
            "max_cluster_count": max_cluster_count,
            "max_cluster_ratio": max_cluster_ratio,
        })

    ec_cluster_stats_df = pd.DataFrame(ec_cluster_stats).sort_values(
        by=["max_cluster_ratio", "num_samples"], ascending=[False, False]
    )
    print(f"Saving per-EC cluster dispersion stats to {ec_stats_csv} ...")
    ec_cluster_stats_df.to_csv(ec_stats_csv, index=False)

    print("\nEvaluate Reaction Clusters Done.")



def evaluate_test_reaction_clusters(
    kmeans_model_path: str,
    test_csv: str,
    structure_path: str,
    test_output_csv: str,
    max_seq_length: int = 600,
    batch_size: int = 96,
    num_workers: int = 64,
    gpu: str = "cuda:0",
    cluster_stats_csv: str = "test_cluster_ec_stats.csv",
    ec_stats_csv: str = "test_ec_cluster_stats.csv",
    mode: str = "evaluate",
):
    """
    使用“训练集上拟合好的 KMeans 模型”来给测试集打 rxn_cluster_label，并在测试集上用 4 级 EC 评估聚类质量。

    流程:
      1) 从 kmeans_model_path 加载在训练集 RXNFP 空间上拟合好的 KMeans 模型。
      2) 用 EnzymeActiveSiteDataset3 加载 test_csv，filter_incomplete_ec=True，自动过滤掉没有完整 4 级 EC 的样本，并统一生成 canonicalize_rxn_smiles。
      3) 对测试集的 unique canonicalize_rxn_smiles 用 RXNFP 编码得到向量。
      4) 用 KMeans.predict 在 RXNFP 空间内为每条测试反应分配簇 ID，得到 test_df['rxn_cluster_label']（RXN_0000 这种字符串）。
      5) 将带有 rxn_cluster_label 的测试集保存到 test_output_csv。
      6) 调用 evaluate_reaction_clusters(test_output_csv, ...)：
            - 用 4 级 EC 作为真值标签
            - 计算 H / C / V
            - 写出 per-cluster / per-EC 的详细统计。
    """

    # ===================== 0. 基本检查 =====================
    if not os.path.exists(kmeans_model_path):
        raise FileNotFoundError(f"KMeans model file not found: {kmeans_model_path}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not os.path.exists(structure_path):
        raise FileNotFoundError(f"Test structure path not found: {structure_path}")

    # ===================== 1. 加载 KMeans 模型 =====================
    print(f"Loading trained KMeans model from {kmeans_model_path} ...")
    with open(kmeans_model_path, "rb") as f:
        obj = pickle.load(f)

    # 兼容几种保存形式：直接保存 KMeans，或 {'model': kmeans}, {'kmeans': kmeans}
    if isinstance(obj, dict):
        if "model" in obj:
            kmeans = obj["model"]
        elif "kmeans" in obj:
            kmeans = obj["kmeans"]
        else:
            kmeans = obj
    else:
        kmeans = obj

    if not isinstance(kmeans, KMeans):
        raise ValueError(f"Loaded object from {kmeans_model_path} is not a sklearn.cluster.KMeans instance.")

    # ===================== 2. 加载测试集并过滤无 4 级 EC 的样本 =====================
    print("Initializing EnzymeActiveSiteDataset3 to load cleaned TEST data...")
    test_dataset = EnzymeActiveSiteDataset3(
        csv_file=test_csv,
        structure_path=structure_path,
        msa_dir="",                 # deprecated
        dca_dir="",                 # deprecated
        dssp_exec_path="",          # deprecated
        dssp_dir="",                # deprecated
        graphec_radius=0.0,         # deprecated
        protein_max_length=max_seq_length,
        max_msa_sequences=0,        # deprecated
        split="test",
        num_samples=None,
        esm_model_name="esm2_t33_650M_UR50D",
        msa_model_name="esm_msa1b_t12_100M_UR50S",
        num_workers=num_workers,
        preprocessing=None,
        preprocessing_dir=None,
        filter_incomplete_ec=True if mode=="evaluate" else False,
    )

    test_df = test_dataset.dataset_df.copy()
    print(f"[test] loaded cleaned dataset_df with {len(test_df)} samples.")

    # 确认 canonicalize_rxn_smiles 存在
    smiles_col = "canonicalize_rxn_smiles"
    if smiles_col not in test_df.columns:
        raise ValueError(f"Expected column '{smiles_col}' in test dataset_df.")

    # 去掉空的 canonical SMILES
    before = len(test_df)
    test_df[smiles_col] = test_df[smiles_col].fillna("").astype(str).str.strip()
    test_df = test_df[test_df[smiles_col] != ""].reset_index(drop=True)
    after = len(test_df)
    print(f"[test] kept {after}/{before} rows after removing empty canonical SMILES.")

    # 收集测试集 unique canonical SMILES
    smiles_series = test_df[smiles_col]
    unique_smiles = smiles_series.unique().tolist()
    print(f"[test] Found {len(unique_smiles)} unique TEST canonical reactions.")
    if len(unique_smiles) == 0:
        raise ValueError("No non-empty canonical reaction SMILES found in TEST dataset_df.")

    # ===================== 3. 构建 RXNFP 编码器（与训练保持一致） =====================
    print("Loading RXNFP default pretrained model for TEST reactions...")
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_default_model_and_tokenizer()
    model = model.to(device)
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

    # ===================== 4. 对测试 unique SMILES 生成 RXNFP 表示 =====================
    rxn_emb_dict = {}
    print("Generating RXNFP embeddings for TEST reactions...")
    for i in top_tqdm(range(0, len(unique_smiles), batch_size), desc="RXNFP Embedding [TEST]"):
        batch_smi = unique_smiles[i : i + batch_size]
        embeddings = rxnfp_generator.convert_batch(batch_smi)
        for smi, emb in zip(batch_smi, embeddings):
            rxn_emb_dict[smi] = np.asarray(emb, dtype=np.float32)

    X_test = np.stack([rxn_emb_dict[smi] for smi in unique_smiles], axis=0)
    print(f"[test] Embedding matrix shape: {X_test.shape}")

    # 维度检查，防止 RXNFP 版本不一致导致维度对不上
    if hasattr(kmeans, "cluster_centers_"):
        if X_test.shape[1] != kmeans.cluster_centers_.shape[1]:
            raise ValueError(
                f"Dimension mismatch between TEST embeddings ({X_test.shape[1]}) "
                f"and KMeans centers ({kmeans.cluster_centers_.shape[1]}). "
                f"Please make sure RXNFP model is the same as used in training."
            )

    # ===================== 5. 用 KMeans 预测测试集簇 ID =====================
    print("[test] Predicting cluster IDs for TEST reactions using trained KMeans ...")
    test_cluster_ids = kmeans.predict(X_test)

    # SMILES -> cluster label 映射
    smiles_to_label_test = {
        smi: f"RXN_{cid:04d}" for smi, cid in zip(unique_smiles, test_cluster_ids)
    }

    test_df["rxn_cluster_label"] = test_df[smiles_col].map(smiles_to_label_test)

    before_assign = len(test_df)
    test_df = test_df[test_df["rxn_cluster_label"].notna()].reset_index(drop=True)
    after_assign = len(test_df)
    print(
        f"[test] kept {after_assign}/{before_assign} rows after assigning TEST cluster labels."
    )

    # 保存带有 rxn_cluster_label 的测试集 CSV（后续 model-3 也可以直接复用）
    print(f"[test] Saving TEST dataframe with rxn_cluster_label to {test_output_csv} ...")
    test_df.to_csv(test_output_csv, index=False)
    print("[test] Saved.")

    # ===================== 6. 在测试集上用 4 级 EC 评估聚类质量 =====================
    # evaluate_reaction_clusters 内部会自动：
    #   - 解析 valid_ec_list 得到 main_ec
    #   - 解析 rxn_cluster_label -> cluster_id
    #   - 丢掉 main_ec 或 cluster_id 为空的行
    #   - 计算 H / C / V，并写出 per-cluster / per-EC 统计
    print("\n[Eval] Evaluating reaction clusters on TEST set (EC as ground truth) ...")
    if mode == "evaluate":
        evaluate_reaction_clusters(
            input_csv=test_output_csv,
            cluster_stats_csv=cluster_stats_csv,
            ec_stats_csv=ec_stats_csv,
        )




