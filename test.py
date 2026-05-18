#!/usr/bin/env python3
"""
EvoSite Testing Script
"""

import os
import sys
import yaml
import argparse
import logging
import time
import warnings
import traceback
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from collections import defaultdict
        
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
import pickle
from functools import partial
import pandas as pd
from pandarallel import pandarallel
import json
from sklearn.metrics import classification_report
# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Import project modules
from model.EvoSite import (EvoSite, EvoSiteCombined, ESMResidueClassifier, MSATransformerClassifier, 
                           EvoSiteBeta, EvoSiteGamma, EvoSitePhiGnet, EvoSiteBetaPhiGnet, EvoSiteDelta, 
                           ECEmbeddingNet, ActiveSiteEmbeddingNet, StackingC, ClusterSite, ResidueEncoderBeta)
from data_loaders.enzyme_msa_dataloaders import (EnzymeMSADataset, create_enzyme_msa_dataloader, 
                                                 EnzymeMSAStructureDataset, collate_enzyme_batch,
                                                 EnzymeActiveSiteDataset3,
                                                 ResidueFeatureDataset, collate_residue_features)

from common.utils import (
    set_seed, setup_logging,
    compute_binary_classification_metrics, compute_multiple_classification_metrics,
    create_defaultdict, process_row,
    read_model_state, move_batch_to_device,
    evaluate_site_type_prediction, evaluate_ec_prediction, build_data_pools, encode_residue_feature, evaluate_site_type_prediction2, format_prediction_to_string
)

current_dir = os.path.dirname(os.path.abspath(__file__)) # /.../ClusterSite
target_tmp_dir = os.path.join(os.path.dirname(current_dir), "tmp") # /.../tmp
os.makedirs(target_tmp_dir, exist_ok=True)
os.environ['TMPDIR'] = target_tmp_dir
current_strategy = torch.multiprocessing.get_sharing_strategy()
if current_strategy != 'file_system':
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f"[System] Shared memory strategy set to: file_system")
    print(f"[System] Temporary files location set to: {target_tmp_dir}")
    
class EvoSiteTester:
    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        """
        Initialize tester with configuration and model checkpoint.
        
        Args:
            config: Inference configuration dictionary
            checkpoint_path: Path to trained model checkpoint
        """
        self.config = config
        self.model_config = config['model']
        self.test_config = config['test']
        self.checkpoint_path = checkpoint_path
        
        # Setup logging first
        self.logger = setup_logging()
        
        # Setup device and reproducibility
        self._device = torch.device(self.test_config['device'])
        if self._device.type == 'cuda':
            # Check if the specified device is available
            if self._device.index is not None and self._device.index < torch.cuda.device_count():
                torch.cuda.set_device(self._device.index)
            else:
                # Fallback to first available GPU if specified device is not available
                self._device = torch.device('cuda:0')
                torch.cuda.set_device(0)
                self.logger.warning(f"Specified device {self.test_config['device']} not available, using cuda:0 instead")
        set_seed(self.test_config['seed'])
        
        # Create output directory
        self.output_dir = self.test_config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize mixed precision
        self.use_amp = self.test_config.get('mixed_precision', False)
        
        # Initialize model and data
        self._setup_model()
        self._setup_data()
        

        
    def _setup_model(self):
        self.logger.info("Loading EvoSite model...")
        
        # Add device to model config
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        # Initialize model
        self.model = EvoSite(**model_config)
        
        # Load the model weights
        training_metadata = self.load_checkpoint(self.checkpoint_path)
            
        self.model = self.model.to(self._device)
        self.model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model loaded successfully from {self.checkpoint_path}")
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log training metadata if available
        if training_metadata:
            if 'epoch' in training_metadata:
                self.logger.info(f"Model was trained for {training_metadata['epoch']} epochs")
            best_metric = training_metadata.get('best_metric') or training_metadata.get('best_val_metric')
            if best_metric is not None:
                self.logger.info(f"Best validation metric: {best_metric:.4f}")
        
        self.logger.info("Maximum memory optimized checkpoint loading completed")
        


    def _setup_data(self):
        """Setup test dataset and data loader."""
        self.logger.info("Setting up test dataset...")
        
        # Test dataset
        self.test_dataset = EnzymeMSADataset(
            csv_file=self.test_config['test_data'],
            msa_dir=self.test_config['msa_dir'],
            dca_dir=self.test_config['dca_dir'],
            max_seq_length=self.test_config['max_seq_length'],
            max_msa_sequences=self.test_config['max_msa_sequences'],
            split="test",
            esm_model_name=self.model_config['esm_model_name'],
            msa_model_name=self.model_config['msa_model_name'],
            test_num_samples=self.test_config.get('test_num_samples', None)
        )
        
        # Data loader
        self.test_loader = create_enzyme_msa_dataloader(
            self.test_dataset,
            batch_size=self.test_config['batch_size'],
            shuffle=False,
            num_workers=self.test_config['num_workers'],
            pin_memory=self.test_config['pin_memory'],
            drop_last=False
        )
        
        self.logger.info(f"Test samples: {len(self.test_dataset)}")
        self.logger.info(f"Test batches: {len(self.test_loader)}")
    


    def evaluate(self, predictions, probabilities, labels, valid_masks=None, residue_counts=None) -> Dict[str, List[float]]:
        """
        Evaluate predictions for one batch data.
        
        Args:
            predictions: Predicted labels [batch_size, seq_len] or [num_samples]
            probabilities: Prediction probabilities [batch_size, seq_len, num_classes] or [num_samples, num_classes]
            labels: True labels [batch_size, seq_len] or [num_samples]
            valid_masks: Mask indicating valid positions [batch_size, seq_len] or None
            residue_counts: List of residue counts for each sequence or None
            
        Returns:
            Dictionary of evaluation metrics, where each key is a metric name and value is a list of per-sample metrics
        """
        self.logger.info("Evaluating...")
        
        if predictions.numel() == 0:
            self.logger.warning("No predictions to evaluate")
            return {}
        
        num_classes = self.model_config['classification_output_dim']
        metrics = {}
        
        if num_classes > 2:
            # 1. 计算二分类指标（活性位点 vs 背景）
            binary_labels = (labels != 0).float()  
            binary_preds = (predictions != 0).float()           
            if probabilities.dim() == 3 and probabilities.size(-1) > 1:
                binary_probs = 1 - probabilities[:, :, 0]  
            else:
                raise ValueError("probabilities must be a 3D tensor with shape [batch_size, seq_len, num_classes]")
            
            binary_metrics = compute_binary_classification_metrics(
                predictions=binary_preds,
                probabilities=binary_probs,
                labels=binary_labels,
                valid_mask=valid_masks,
                metrics=['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc', 'ap', 'mcc']
            )
            
            for metric_name, metric_values in binary_metrics.items():
                metrics[f"binary_{metric_name}"] = metric_values
            
            # 2. 为每个位点类型单独计算二分类指标
            for site_type in range(0, num_classes):  
                type_labels = (labels == site_type).float()             # [batch_size, seq_len]
                type_preds = (predictions == site_type).float()         # [batch_size, seq_len]
                if probabilities.dim() == 3 and probabilities.size(-1) > site_type:
                    type_probs = probabilities[:, :, site_type]         # [batch_size, seq_len]
                else:
                    type_probs = None
                
                if type_labels.sum() == 0:
                    continue
                
                site_metrics = compute_binary_classification_metrics(
                    predictions=type_preds,
                    probabilities=type_probs,
                    labels=type_labels,
                    valid_mask=valid_masks,
                    metrics=['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc', 'ap', 'mcc']
                )
                
                for metric_name, metric_values in site_metrics.items():
                    metrics[f"pre_type_{site_type}_{metric_name}"] = metric_values
            
            # 3. 计算多分类指标
            multi_metrics = compute_multiple_classification_metrics(
                pred=predictions,
                gt=labels,
                num_site_types=num_classes,
                valid_mask=valid_masks
            )

            for metric_name, values in multi_metrics.items():
                if values:  
                    metric_key = metric_name.replace('-', '_')
                    metrics[metric_key] = values
                    
        else:
            # 二分类情况
            binary_metrics = compute_binary_classification_metrics(
                predictions=predictions,
                probabilities=probabilities[:, 1] if probabilities.dim() > 1 and probabilities.size(-1) > 1 else probabilities,
                labels=labels,
                valid_mask=valid_masks,
                metrics=['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc', 'ap', 'mcc']
            )
            
            for metric_name, metric_values in binary_metrics.items():
                metrics[f"binary_{metric_name}"] = metric_values
        
        return metrics



    def run_inference(self) -> Dict[str, Any]:

        self.logger.info("Starting testing...")
        start_time = time.time()
        all_metrics = defaultdict(list)

        pbar = tqdm(self.test_loader, desc="testing", unit="batch")

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Filter out any batches that failed during collation.
                    if batch is None:
                        self.logger.warning(f"Skipping empty or failed batch at index {batch_idx}.")
                        raise ValueError("Batch is None")
                    
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(batch)
                    
                    logits = outputs['logits']  # [batch_size, seq_len, num_classes]
                    labels = batch['labels']    # [batch_size, seq_len]
                    
                    if 'esm_valid_tokens_mask' in outputs:
                        valid_mask = outputs['esm_valid_tokens_mask']
                    else:
                        print("No valid tokens mask provided")
                        valid_mask = torch.ones_like(labels, dtype=torch.bool)      # 如果不提供掩码，默认所有位置都是有效的
                    
                    probs = torch.softmax(logits, dim=-1)  # [batch_size, seq_len, num_classes]
                    preds = torch.argmax(probs, dim=-1)    # [batch_size, seq_len]
                    
                    batch_predictions = preds
                    batch_probabilities = probs
                    batch_labels = labels
                    batch_valid_masks = valid_mask
                    batch_residue_counts = batch_valid_masks.sum(dim=1).tolist()
                    
                    # Compute metrics for current batch
                    if self.test_config.get('compute_metrics', True) and batch_predictions.numel() > 0:
                        batch_metrics = self.evaluate(
                            batch_predictions, 
                            batch_probabilities, 
                            batch_labels, 
                            batch_valid_masks, 
                            batch_residue_counts
                        )
                        
                        self.logger.info(f"Batch {batch_idx} metrics:")
                
                        for metric_name, values in batch_metrics.items():
                            if metric_name not in all_metrics:
                                all_metrics[metric_name] = []
                            all_metrics[metric_name].extend(values)
                            
                            # 计算并显示当前批次的平均值（仅用于日志）
                            if values:
                                avg_value = sum(values) / len(values)
                                self.logger.info(f"  {metric_name}: {avg_value:.4f} (avg of {len(values)} samples)")
                    
                except Exception as e:
                    sample_ids = batch.get('sample_id', ['unknown'] * len(batch.get('protein_ids', ['unknown'])))
                    protein_ids = batch.get('protein_ids', ['unknown'] * len(sample_ids))
                    self.logger.warning(f"Error processing batch {batch_idx} - skipping batch")
                    self.logger.warning(f"Error type: {type(e).__name__}")
                    self.logger.warning(f"Error message: {str(e)}")
                    self.logger.warning(f"Problem samples: {list(zip(sample_ids, protein_ids))}")
                    self.logger.warning(f"Batch size: {len(sample_ids)}")
                    continue
        
        pbar.close()
        
        # 计算平均指标
        if all_metrics:
            final_metrics = {}
            for metric_name, values in all_metrics.items():
                if values:
                    final_metrics[metric_name] = sum(values) / len(values)
            
            self.logger.info("\nOverall metrics (averaged across all samples):")
            for metric_name, value in sorted(final_metrics.items()):
                self.logger.info(f"  {metric_name}: {value:.4f}")
        
        total_time = time.time() - start_time
        self.logger.info(f"Inference completed in {total_time:.2f} seconds")
        


    def _move_batch_to_device(self, batch: Any) -> Any:
        """
        Recursively move any nested container of tensors or compatible objects
        to the specified device.

        This unified method is designed to be robust for various data structures,
        including simple tensors, dictionaries, lists, and complex objects from
        libraries like DGL, TorchDrug, or HuggingFace Transformers.

        It handles:
        - torch.Tensor.
        - Custom objects with a callable `.to(device)` method (duck-typing).
        - Nested dictionaries, lists, and tuples, preserving the container type.
        - Automatic conversion of NumPy arrays to tensors.
        - Primitives (str, int, float, bool, None) are returned unmodified.
        
        Args:
            batch: A batch of data, which can be a tensor or a nested container.
            
        Returns:
            The batch with all movable components transferred to `self._device`.
        """
        # Fast path for primitive types and None, which don't need moving.
        if batch is None or isinstance(batch, (str, bytes, int, float, bool)):
            return batch

        # The most common case: a torch.Tensor.
        # Using isinstance is a slight performance optimization over hasattr for this frequent check.
        if isinstance(batch, torch.Tensor):
            try:
                # Use non_blocking for performance gains with pinned memory.
                return batch.to(self._device, non_blocking=True)
            except TypeError:
                # Fallback for older PyTorch versions or specific tensor types
                # that might not support non_blocking.
                return batch.to(self._device)

        # Generic handling for any object that implements a `.to()` method (duck-typing).
        # This covers DGL graphs, TorchDrug's PackedGraph, HuggingFace's BatchEncoding, etc.
        # We also ensure it's not a Module, which should not be moved this way.
        if hasattr(batch, "to") and callable(getattr(batch, "to")) and not isinstance(batch, nn.Module):
            try:
                # EAFP (Easier to Ask for Forgiveness than Permission) approach.
                return batch.to(self._device, non_blocking=True)
            except TypeError:
                # If the '.to' method doesn't support 'non_blocking', try without it.
                return batch.to(self._device)
            except Exception:
                # If the '.to' method fails for any other reason, return the object as is.
                return batch

        # Recursively handle dictionaries, preserving the dictionary type.
        if isinstance(batch, dict):
            return type(batch)({k: self._move_batch_to_device(v) for k, v in batch.items()})
        
        # Recursively handle lists and tuples, preserving the container type.
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(item) for item in batch)

        # Handle NumPy arrays, assuming 'import numpy as np' is present at the top of the file.
        # This check is placed after the main tensor/object checks for optimization.
        if isinstance(batch, np.ndarray):
            return torch.from_numpy(batch).to(self._device, non_blocking=True)
        
        # If the object type is not recognized, return it unmodified.
        return batch



    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint with modular structure and backward compatibility.
        Integrates utils.py functionality with memory optimization and DDP handling.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        
        # Detect checkpoint format
        is_modular = ('model_state' in checkpoint and 'training_state' in checkpoint)
        
        training_metadata = {}
        
        if is_modular:
            # Load modular checkpoint format
            model_state_dict = checkpoint['model_state']['state_dict']
            
            # Extract training metadata
            training_state = checkpoint['training_state']
            training_metadata['epoch'] = training_state['epoch']
            training_metadata['global_step'] = training_state.get('global_step', 0)
            training_metadata['best_val_metric'] = training_state.get('best_val_metric', 0.0)
            training_metadata['patience_counter'] = training_state.get('patience_counter', 0)
            
        else:
            model_state_dict = checkpoint['model_state_dict']
            
            training_metadata['epoch'] = checkpoint.get('epoch', 0)
            training_metadata['best_metric'] = checkpoint.get('best_metric', 0.0)
            training_metadata['global_step'] = checkpoint.get('global_step', 0)
        
        # Handle DDP prefix mismatch for model loading
        current_model_keys = list(self.model.state_dict().keys())
        checkpoint_keys = list(model_state_dict.keys())
        
        if len(current_model_keys) > 0 and len(checkpoint_keys) > 0:
            current_has_module = current_model_keys[0].startswith('module.')
            checkpoint_has_module = checkpoint_keys[0].startswith('module.')
            
            # Add module prefix if current model has it but checkpoint doesn't
            if current_has_module and not checkpoint_has_module:
                model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
            # Remove module prefix if current model doesn't have it but checkpoint does
            elif not current_has_module and checkpoint_has_module:
                model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        
        # Load model state
        self.model.load_state_dict(model_state_dict)
        
        # Memory cleanup - release checkpoint data
        del checkpoint, model_state_dict
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return training_metadata



class ESM_MSATransformer_AblationTester(EvoSiteTester):
    """
    Unified tester for ablation study models (ESM and MSA baselines).
    """
    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        """
        Initialize ablation tester with configuration and model checkpoint.
        
        Args:
            config: Inference configuration dictionary
            checkpoint_path: Path to trained model checkpoint
        """
        self.model_type = config.get('model', {}).get('model_type', 'esm').lower()
        super().__init__(config, checkpoint_path)



    def _setup_model(self):
        """Setup ablation model based on configuration."""
        self.logger.info(f"Loading {self.model_type.upper()} ablation model...")

        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        # Create ablation model based on type
        if self.model_type.lower() == 'esm':
            self.model = ESMResidueClassifier(
                esm_model_name=model_config.get('esm_model_name', "esm2_t33_650M_UR50D"),
                num_classes=model_config.get('classification_output_dim', 4),
                hidden_dim_multiplier=model_config.get('hidden_dim_multiplier', 2),
                dropout=model_config.get('dropout', 0.1),
                device=self._device
            )
        elif self.model_type.lower() == 'msa':
            self.model = MSATransformerClassifier(
                msa_model_name=model_config.get('msa_model_name', "esm_msa1b_t12_100M_UR50S"),
                num_classes=model_config.get('classification_output_dim', 4),
                hidden_dim_multiplier=model_config.get('hidden_dim_multiplier', 2),
                dropout=model_config.get('dropout', 0.1),
                device=self._device
            )
        else:
            raise ValueError(f"Unsupported ablation model type: {self.model_type}. Supported: 'esm', 'msa'")
        
        training_metadata = self.load_checkpoint(self.checkpoint_path)
        self.model = self.model.to(self._device)
        self.model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"{self.model_type.upper()} model loaded successfully from {self.checkpoint_path}")
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log training metadata if available
        if training_metadata:
            if 'epoch' in training_metadata:
                self.logger.info(f"Model was trained for {training_metadata['epoch']} epochs")
            best_metric = training_metadata.get('best_metric') or training_metadata.get('best_val_metric')
            if best_metric is not None:
                self.logger.info(f"Best validation metric: {best_metric:.4f}")



    def _setup_data(self):
        """Setup test dataset and data loader for ablation models."""
        self.logger.info("Setting up test dataset...")
        
        # For ablation models, we use simplified model names
        # Always get both model names from config for dataset compatibility
        esm_model_name = self.model_config.get('esm_model_name', 'esm2_t33_650M_UR50D')
        msa_model_name = self.model_config.get('msa_model_name', 'esm_msa1b_t12_100M_UR50S')
        
        # Test dataset
        self.test_dataset = EnzymeMSADataset(
            csv_file=self.test_config['test_data'],
            msa_dir=self.test_config['msa_dir'],
            dca_dir=self.test_config['dca_dir'],
            max_seq_length=self.test_config['max_seq_length'],
            max_msa_sequences=self.test_config['max_msa_sequences'],
            split="test",
            esm_model_name=esm_model_name,
            msa_model_name=msa_model_name,
            test_num_samples=self.test_config.get('test_num_samples', None)
        )
        
        # Data loader
        self.test_loader = create_enzyme_msa_dataloader(
            self.test_dataset,
            batch_size=self.test_config['batch_size'],
            shuffle=False,
            num_workers=self.test_config['num_workers'],
            pin_memory=self.test_config['pin_memory'],
            drop_last=False
        )
        
        self.logger.info(f"Test samples: {len(self.test_dataset)}")
        self.logger.info(f"Test batches: {len(self.test_loader)}")



    def _forward_ablation_model(self, batch: Dict[str, Union[torch.Tensor, List[torch.Tensor], List[str], List[int]]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ablation models with proper parameter extraction.
        
        Args:
            batch: Batch dictionary from data loader
            
        Returns:
            Model outputs dictionary with 'logits' and 'valid_mask' keys
        """
        # Generate valid_mask for tokens (ESM-1b style)
        # Based on collate_enzyme_msa_batch token configuration
        padding_idx = 1  # <pad> token index
        cls_idx = 0      # <cls> token index (used as BOS)
        eos_idx = 2      # <eos> token index
        
        if self.model_type == 'esm':
            # ESM baseline model expects: tokens and valid_mask
            if 'sequences' not in batch:
                raise ValueError("ESM model requires 'sequences' in batch")
            
            tokens = batch['sequences']  # [batch_size, seq_len]
            
            # Generate valid_mask: True for valid amino acid tokens, False for special tokens
            valid_mask = torch.ones_like(tokens, dtype=torch.bool)
            valid_mask &= (tokens != padding_idx)  # Not padding
            valid_mask &= (tokens != cls_idx)      # Not CLS/BOS
            valid_mask &= (tokens != eos_idx)      # Not EOS
            
            outputs = self.model(tokens, valid_mask)
            
        elif self.model_type == 'msa':
            # MSA baseline model expects: msa_tokens and valid_mask
            if 'msa' not in batch:
                raise ValueError("MSA model requires 'msa' in batch")
            
            msa_tokens = batch['msa']  # [batch_size, num_seqs, seq_len]
            
            # Generate valid_mask based on the first sequence (query sequence)
            # Extract the first sequence from MSA for valid_mask calculation
            query_tokens = msa_tokens[:, 0, :]  # [batch_size, seq_len]
            valid_mask = torch.ones_like(query_tokens, dtype=torch.bool)
            valid_mask &= (query_tokens != padding_idx)  # Not padding
            valid_mask &= (query_tokens != cls_idx)      # Not CLS/BOS
            valid_mask &= (query_tokens != eos_idx)      # Not EOS
            
            outputs = self.model(msa_tokens, valid_mask)
            
        else:
            raise ValueError(f"Unsupported ablation model type: {self.model_type}")
        
        return outputs



    def run_inference(self) -> Dict[str, Any]:
        """Run inference on test data using ablation models."""
        self.logger.info(f"Starting {self.model_type.upper()} ablation testing...")
        start_time = time.time()
        all_metrics = defaultdict(list)

        # Progress bar
        pbar = tqdm(
            self.test_loader,
            desc=f"{self.model_type.upper()} testing",
            unit="batch"
        )
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                try:
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass through ablation model
                    with autocast(enabled=self.use_amp):
                        outputs = self._forward_ablation_model(batch)
                    
                    logits = outputs['logits']  # [batch_size, seq_len, num_classes]
                    labels = batch['labels']    # [batch_size, seq_len]
                    
                    if 'valid_mask' in outputs:
                        valid_mask = outputs['valid_mask']
                    else:
                        self.logger.warning("No valid tokens mask provided by ablation model")
                        valid_mask = torch.ones_like(labels, dtype=torch.bool)      # 如果不提供掩码，默认所有位置都是有效的
                    
                    probs = torch.softmax(logits, dim=-1)  # [batch_size, seq_len, num_classes]
                    preds = torch.argmax(probs, dim=-1)    # [batch_size, seq_len]
                    
                    batch_predictions = preds
                    batch_probabilities = probs
                    batch_labels = labels
                    batch_valid_masks = valid_mask
                    batch_residue_counts = batch_valid_masks.sum(dim=1).tolist()
                    
                    # Compute metrics for current batch
                    if self.test_config.get('compute_metrics', True) and batch_predictions.numel() > 0:
                        batch_metrics = self.evaluate(
                            batch_predictions, 
                            batch_probabilities, 
                            batch_labels, 
                            batch_valid_masks, 
                            batch_residue_counts
                        )
                        
                        self.logger.info(f"Batch {batch_idx} metrics:")
                
                        for metric_name, values in batch_metrics.items():
                            if metric_name not in all_metrics:
                                all_metrics[metric_name] = []
                            all_metrics[metric_name].extend(values)
                            
                            # 计算并显示当前批次的平均值（仅用于日志）
                            if values:
                                avg_value = sum(values) / len(values)
                                self.logger.info(f"  {metric_name}: {avg_value:.4f} (avg of {len(values)} samples)")
                    
                except Exception as e:
                    sample_ids = batch.get('sample_id', ['unknown'] * len(batch.get('protein_ids', ['unknown'])))
                    protein_ids = batch.get('protein_ids', ['unknown'] * len(sample_ids))
                    self.logger.warning(f"Error processing batch {batch_idx} - skipping batch")
                    self.logger.warning(f"Error type: {type(e).__name__}")
                    self.logger.warning(f"Error message: {str(e)}")
                    self.logger.warning(f"Problem samples: {list(zip(sample_ids, protein_ids))}")
                    self.logger.warning(f"Batch size: {len(sample_ids)}")
                    continue
        
        pbar.close()
        
        # 计算平均指标
        if all_metrics:
            final_metrics = {}
            for metric_name, values in all_metrics.items():
                if values:
                    final_metrics[metric_name] = sum(values) / len(values)
            
            self.logger.info(f"\n{self.model_type.upper()} Overall metrics (averaged across all samples):")
            for metric_name, value in sorted(final_metrics.items()):
                self.logger.info(f"  {metric_name}: {value:.4f}")
        
        total_time = time.time() - start_time
        self.logger.info(f"{self.model_type.upper()} inference completed in {total_time:.2f} seconds")

    
    
class EvoSiteCombinedTester(EvoSiteTester):
    """
    Tester for EvoSiteCombined model.
    """
    
    def _setup_model(self):
        """Load the EvoSiteCombined model from checkpoint."""
        if self.logger:
            self.logger.info("Loading EvoSiteCombined model...")
        
        # Add device to model config
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        # Initialize combined model
        self.model = EvoSiteCombined(**model_config)
        
        # Load checkpoint with maximum memory optimization for inference
        training_metadata = self.load_checkpoint(self.checkpoint_path)
            
        self.model = self.model.to(self._device)
        self.model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"EvoSiteCombined loaded successfully from {self.checkpoint_path}")
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        self.logger.info(f"Combined mode features:")
        self.logger.info(f"  - Bias injection: {self.model.use_relative_bias_injection}")
        self.logger.info(f"  - Auxiliary losses: {self.model.use_auxiliary_losses}")
        self.logger.info(f"  - Use concat map: {self.model.use_concat_map}")
        
        # Log training metadata if available
        if training_metadata:
            if 'epoch' in training_metadata:
                self.logger.info(f"Model was trained for {training_metadata['epoch']} epochs")
            best_metric = training_metadata.get('best_metric') or training_metadata.get('best_val_metric')
            if best_metric is not None:
                self.logger.info(f"Best validation metric: {best_metric:.4f}")
        
        self.logger.info("Maximum memory optimized checkpoint loading completed")



class EvoSiteBetaTester(EvoSiteTester):
    """
    Tester for EvoSiteBeta model.
    """
    
    def _setup_model(self):
        self.logger.info("Loading EvoSiteBeta model...")
        
        # Add the target device to the model configuration.
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        # Instantiate the EvoSiteBeta model.
        self.model = EvoSiteBeta(**model_config)
        
        # Load the model weights
        training_metadata = self.load_checkpoint(self.checkpoint_path)
            
        self.model = self.model.to(self._device)
        self.model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model loaded successfully from {self.checkpoint_path}")
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log training metadata if available
        if training_metadata:
            if 'epoch' in training_metadata:
                self.logger.info(f"Model was trained for {training_metadata['epoch']} epochs")
            best_metric = training_metadata.get('best_metric') or training_metadata.get('best_val_metric')
            if best_metric is not None:
                self.logger.info(f"Best validation metric during training: {best_metric:.4f}")
        
        self.logger.info("EvoSiteBeta model setup completed.")



    def _setup_data(self):
        """Setup the test dataset and DataLoader for EvoSiteBeta."""
        self.logger.info("Setting up test dataset for EvoSiteBeta...")

        self.test_dataset = EnzymeMSAStructureDataset(
            csv_file=self.test_config['test_data'],
            structure_path=self.test_config['structure_path'],
            msa_dir=self.test_config['msa_dir'],
            dca_dir=self.test_config['dca_dir'],
            dssp_exec_path=self.test_config.get('dssp_exec_path'),
            dssp_dir=self.test_config.get('dssp_dir'),
            graphec_radius=self.test_config.get('graphec_radius'),
            protein_max_length=self.test_config['max_seq_length'],
            max_msa_sequences=self.test_config['max_msa_sequences'],
            split="test",
            num_samples=self.test_config.get('test_num_samples', None),
            esm_model_name=self.test_config['esm_model_name'],
            msa_model_name=self.test_config['msa_model_name'],
            num_workers=self.test_config['num_workers'],
            preprocessing=self.test_config.get('preprocessing', 'load'),
            preprocessing_dir=self.test_config.get('test_preprocessing_dir', None)
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_config['batch_size'],
            shuffle=False,
            num_workers=self.test_config['num_workers'],
            pin_memory=self.test_config['pin_memory'],
            collate_fn=collate_enzyme_batch,
            drop_last=False
        )
        
        self.logger.info(f"Test samples: {len(self.test_dataset)}")
        self.logger.info(f"Test batches: {len(self.test_loader)}")



    def run_inference(self) -> Dict[str, Any]:

        self.logger.info("Starting EvoSiteBeta testing...")
        start_time = time.time()
        all_metrics = defaultdict(list)

        pbar = tqdm(self.test_loader, desc="EvoSiteBeta Testing", unit="batch")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Filter out any batches that failed during collation.
                    if batch is None:
                        self.logger.warning(f"Skipping empty or failed batch at index {batch_idx}.")
                        raise ValueError("Batch is None")
                        
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass
                    with autocast(enabled=self.use_amp):
                        # EvoSiteBeta returns unpacked features for valid residues.
                        flat_logits, protein_mask = self.model(batch)

                    flat_labels = batch['targets'].long()
                    batch_size, max_len = protein_mask.shape
                    num_classes = self.model_config['classification_output_dim']

                    padded_labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=self._device)
                    padded_logits = torch.zeros(batch_size, max_len, num_classes, device=self._device)
                    padded_labels[protein_mask.bool()] = flat_labels
                    padded_logits[protein_mask.bool()] = flat_logits
                    valid_mask = protein_mask
                    
                    # --- Metric Calculation ---
                    probs = torch.softmax(padded_logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1)

                    if self.test_config.get('compute_metrics', True) and preds.numel() > 0:
                        batch_metrics = self.evaluate(
                            predictions=preds, 
                            probabilities=probs, 
                            labels=padded_labels, 
                            valid_masks=valid_mask
                        )
                        
                        self.logger.info(f"Batch {batch_idx} metrics:")

                        for metric_name, values in batch_metrics.items():
                            if metric_name not in all_metrics:
                                all_metrics[metric_name] = []
                            all_metrics[metric_name].extend(values)

                            # 计算并显示当前批次的平均值（仅用于日志）
                            if values:
                                avg_value = sum(values) / len(values)
                                self.logger.info(f"  {metric_name}: {avg_value:.4f} (avg of {len(values)} samples)")
                    
                except Exception as e:
                    uniprot_ids = batch.get('uniprot_ids', ['N/A'])
                    self.logger.error(f"Error processing batch {batch_idx}. Skipping. Error: {e}", exc_info=True)
                    self.logger.error(f"Problematic UniProt IDs might be in: {uniprot_ids}")
                    continue
        
        pbar.close()
        
        # 计算平均指标
        if all_metrics:
            final_metrics = {}
            for metric_name, values in all_metrics.items():
                if values:
                    final_metrics[metric_name] = sum(values) / len(values)
            
            self.logger.info("\nOverall metrics (averaged across all samples):")
            for metric_name, value in sorted(final_metrics.items()):
                self.logger.info(f"  {metric_name}: {value:.4f}")
        
        total_time = time.time() - start_time
        self.logger.info(f"Inference completed in {total_time:.2f} seconds")
    


class EvoSiteGammaTester(EvoSiteBetaTester):
    """
    Tester for the EvoSiteGamma model.
    Inherits most functionality from EvoSiteBetaTester, overriding only the
    model setup to instantiate the correct EvoSiteGamma model. The evaluation
    logic remains the same as inference mode does not involve auxiliary heads.
    """
    
    def _setup_model(self):
        """Override to load the EvoSiteGamma model from a checkpoint."""
        self.logger.info("Loading EvoSiteGamma model...")
        
        # Add the target device to the model configuration.
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        # Instantiate the EvoSiteGamma model.
        self.model = EvoSiteGamma(**model_config)
        
        # Load the model weights from the checkpoint.
        training_metadata = self.load_checkpoint(self.checkpoint_path)
            
        self.model = self.model.to(self._device)
        self.model.eval()  # Set model to evaluation mode
        
        # Log model information.
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"EvoSiteGamma model loaded successfully from {self.checkpoint_path}")
        self.logger.info(f"Total model parameters: {total_params:,}")
        
        # Log relevant training metadata if available.
        if training_metadata:
            if 'epoch' in training_metadata:
                self.logger.info(f"Model was trained for {training_metadata['epoch']} epochs")
            best_metric = training_metadata.get('best_val_metric')
            if best_metric is not None:
                self.logger.info(f"Best validation metric during training: {best_metric:.4f}")
        
        self.logger.info("EvoSiteGamma model setup completed for testing.")



class EvoSitePhiGnetTester(EvoSiteBetaTester):

    def _setup_model(self):
        self.logger.info("Loading EvoSitePhiGnet model...")
        
        # Add the target device to the model configuration.
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        # Instantiate the EvoSitePhiGnet model.
        self.model = EvoSitePhiGnet(**model_config)
        
        # Load the model weights
        training_metadata = self.load_checkpoint(self.checkpoint_path)
            
        self.model = self.model.to(self._device)
        self.model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model loaded successfully from {self.checkpoint_path}")
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log training metadata if available
        if training_metadata:
            if 'epoch' in training_metadata:
                self.logger.info(f"Model was trained for {training_metadata['epoch']} epochs")
            best_metric = training_metadata.get('best_metric') or training_metadata.get('best_val_metric')
            if best_metric is not None:
                self.logger.info(f"Best validation metric during training: {best_metric:.4f}")
        
        self.logger.info("EvoSitePhiGnet model setup completed for testing.")



class EvoSiteBetaPhiGnetTester(EvoSiteBetaTester):
    def _setup_model(self):
        self.logger.info("Loading EvoSiteBetaPhiGnet model...")
        
        # Add the target device to the model configuration.
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        # Instantiate the EvoSiteBetaPhiGnet model.
        self.model = EvoSiteBetaPhiGnet(**model_config)
        
        # Load the model weights from the specified checkpoint.
        training_metadata = self.load_checkpoint(self.checkpoint_path)
            
        self.model = self.model.to(self._device)
        self.model.eval()  # Set model to evaluation mode.
        
        # Log model information.
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model loaded successfully from {self.checkpoint_path}")
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log relevant training metadata if available from the checkpoint.
        if training_metadata:
            if 'epoch' in training_metadata:
                self.logger.info(f"Model was trained for {training_metadata['epoch']} epochs")
            best_metric = training_metadata.get('best_metric') or training_metadata.get('best_val_metric')
            if best_metric is not None:
                self.logger.info(f"Best validation metric during training: {best_metric:.4f}")
        
        self.logger.info("EvoSiteBetaPhiGnet model setup completed for testing.")



class EvoSiteDeltaTester(EvoSiteBetaTester):
    def _setup_model(self):
        self.logger.info("Loading EvoSiteDelta model...")
        
        # Add the target device to the model configuration.
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        # Instantiate the EvoSiteDelta model.
        self.model = EvoSiteDelta(**model_config)
        
        # Load the model weights
        training_metadata = self.load_checkpoint(self.checkpoint_path)
            
        self.model = self.model.to(self._device)
        self.model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"EvoSiteDelta model loaded successfully from {self.checkpoint_path}")
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log training metadata if available
        if training_metadata:
            if 'epoch' in training_metadata:
                self.logger.info(f"Model was trained for {training_metadata['epoch']} epochs")
            best_metric = training_metadata.get('best_metric') or training_metadata.get('best_val_metric')
            if best_metric is not None:
                self.logger.info(f"Best validation metric during training: {best_metric:.4f}")
        
        self.logger.info("EvoSiteDelta model setup completed for testing.")


# class SiteSupConModelTester(EvoSiteBetaTester):
#     def __init__(self, config: Dict[str, Any], checkpoint_path: str):
#         """
#         Initialize tester with configuration and model checkpoint.
        
#         Args:
#             config: Inference configuration dictionary
#             checkpoint_path: Path to trained model checkpoint
#         """
#         self.config = config
#         self.model_config = config['model']
#         self.test_config = config['common']
#         self.checkpoint_path = checkpoint_path
        
#         # Setup logging first
#         self.logger = setup_logging()
        
#         # Setup device and reproducibility
#         self._device = torch.device(self.test_config['device'])
#         if self._device.type == 'cuda':
#             # Check if the specified device is available
#             if self._device.index is not None and self._device.index < torch.cuda.device_count():
#                 torch.cuda.set_device(self._device.index)
#             else:
#                 # Fallback to first available GPU if specified device is not available
#                 self._device = torch.device('cuda:0')
#                 torch.cuda.set_device(0)
#                 self.logger.warning(f"Specified device {self.test_config['device']} not available, using cuda:0 instead")
#         set_seed(self.test_config['seed'])
        
#         # Create output directory
#         self.output_dir = self.test_config['output_dir']
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         # Initialize mixed precision
#         self.use_amp = self.test_config.get('mixed_precision', False)
        
#         # Initialize model and data
#         self._setup_model()
#         self._setup_data()

#     def _setup_model(self):
#         self.logger.info("Loading SiteSupConModel model...")
        
#         # Add the target device to the model configuration.
#         model_config = self.model_config.copy()
#         model_config['device'] = self._device
        
#         # Instantiate the EvoSiteDelta model.
#         self.model = SiteSupConModel(model_config)
        
#         # Load the model weights
#         training_metadata = self.load_checkpoint(self.checkpoint_path)
            
#         self.model = self.model.to(self._device)
#         self.model.eval()
        
#         # Log model info
#         total_params = sum(p.numel() for p in self.model.parameters())
#         trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
#         self.logger.info(f"EvoSiteDelta model loaded successfully from {self.checkpoint_path}")
#         self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
#         # Log training metadata if available
#         if training_metadata:
#             if 'epoch' in training_metadata:
#                 self.logger.info(f"Model was trained for {training_metadata['epoch']} epochs")
#             best_metric = training_metadata.get('best_metric') or training_metadata.get('best_val_metric')
#             if best_metric is not None:
#                 self.logger.info(f"Best validation metric during training: {best_metric:.4f}")
        
#         self.logger.info("EvoSiteDelta model setup completed for testing.")


#     def _setup_data(self):
#         """Setup the test dataset and DataLoader for EvoSiteBeta."""
#         self.logger.info("Setting up test dataset for EvoSiteBeta...")

#         self.test_dataset = EnzymeActiveSiteDataset3(
#             csv_file=self.config['common']['test_data'],
#             structure_path=self.config['common']['structure_path'],
#             msa_dir='',                 # Deprecated arguments
#             dca_dir='',                 # Deprecated arguments
#             dssp_exec_path='',          # Deprecated arguments
#             dssp_dir='',                # Deprecated arguments
#             graphec_radius=0,           # Deprecated arguments
#             protein_max_length=self.config['common']['max_seq_length'],
#             max_msa_sequences=0,        # Deprecated arguments
#             split='test',
#             num_samples=None,           # Deprecated arguments
#             esm_model_name='esm2_t33_650M_UR50D',
#             msa_model_name='esm_msa1b_t12_100M_UR50S',
#             num_workers=self.config['common']['num_workers'],
#             preprocessing=None,         # Deprecated arguments
#             preprocessing_dir=None,     # Deprecated arguments
#             filter_incomplete_ec=False,
#         )
        
#         self.test_loader = DataLoader(
#             self.test_dataset,
#             batch_size=1,
#             shuffle=False,
#             num_workers=self.test_config['num_workers'],
#             pin_memory=True,
#             collate_fn=collate_enzyme_batch,
#             drop_last=False
#         )
        
#         self.logger.info(f"Test samples: {len(self.test_dataset)}")
#         self.logger.info(f"Test batches: {len(self.test_loader)}")



#     def run_inference(self) -> Dict[str, Any]:

#         self.logger.info("Starting EvoSiteBeta testing...")
#         start_time = time.time()
#         all_metrics = defaultdict(list)

#         pbar = tqdm(self.test_loader, desc="EvoSiteBeta Testing", unit="batch")
        
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(pbar):
#                 try:
#                     # Filter out any batches that failed during collation.
#                     if batch is None:
#                         self.logger.warning(f"Skipping empty or failed batch at index {batch_idx}.")
#                         raise ValueError("Batch is None")
                        
#                     batch = self._move_batch_to_device(batch)
                    
#                     # Forward pass
#                     with autocast(enabled=self.use_amp):
#                         # EvoSiteBeta returns unpacked features for valid residues.
#                         flat_logits, protein_mask = self.model(batch, return_z=False)   # 推理模式：return_z=False

#                     flat_labels = batch['targets'].long()
#                     batch_size, max_len = protein_mask.shape
#                     num_classes = 4

#                     padded_labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=self._device)
#                     padded_logits = torch.zeros(batch_size, max_len, num_classes, device=self._device)
#                     padded_labels[protein_mask.bool()] = flat_labels
#                     padded_logits[protein_mask.bool()] = flat_logits
#                     valid_mask = protein_mask
                    
#                     # --- Metric Calculation ---
#                     probs = torch.softmax(padded_logits, dim=-1)
#                     preds = torch.argmax(probs, dim=-1)

#                     if preds.numel() > 0:
#                         batch_metrics = self.evaluate(
#                             predictions=preds, 
#                             probabilities=probs, 
#                             labels=padded_labels, 
#                             valid_masks=valid_mask
#                         )
                        
#                         self.logger.info(f"Batch {batch_idx} metrics:")

#                         for metric_name, values in batch_metrics.items():
#                             if metric_name not in all_metrics:
#                                 all_metrics[metric_name] = []
#                             all_metrics[metric_name].extend(values)

#                             # 计算并显示当前批次的平均值（仅用于日志）
#                             if values:
#                                 avg_value = sum(values) / len(values)
#                                 self.logger.info(f"  {metric_name}: {avg_value:.4f} (avg of {len(values)} samples)")
                    
#                 except Exception as e:
#                     uniprot_ids = batch.get('uniprot_ids', ['N/A'])
#                     self.logger.error(f"Error processing batch {batch_idx}. Skipping. Error: {e}", exc_info=True)
#                     self.logger.error(f"Problematic UniProt IDs might be in: {uniprot_ids}")
#                     continue
        
#         pbar.close()
        
#         # 计算平均指标
#         if all_metrics:
#             final_metrics = {}
#             for metric_name, values in all_metrics.items():
#                 if values:
#                     final_metrics[metric_name] = sum(values) / len(values)
            
#             self.logger.info("\nOverall metrics (averaged across all samples):")
#             for metric_name, value in sorted(final_metrics.items()):
#                 self.logger.info(f"  {metric_name}: {value:.4f}")
        
#         total_time = time.time() - start_time
#         self.logger.info(f"Inference completed in {total_time:.2f} seconds")


# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------


class ClusterSiteTester:

    def __init__(self, config: dict, args: argparse.Namespace):
        """
        Args:
            config: YAML 总配置字典
            args: 命令行参数，包含各模块的模型路径
        """
        self.config = config
        self.args = args
        self.device = torch.device(config['common']['device'])
        self.use_amp = config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32
        
        self.model = ClusterSite(config['model']).to(self.device)
        self.model.float()
        self.load_checkpoint()
        self.model.eval()


        self.part1_residue_features = {}       # 仅经过模块一编码器后的残基特征 [L, D]
        self.part1_protein_features = {}       # 仅经过模块一编码器后的蛋白质特征（平均池化） [D]
        self.part1_probs = {}                  # 仅经过模块一编码器后的预测概率 (Softmax后) [L, C]
        
        self.part2_ec_residue_features = {}    # 经过模块二 ECEmbeddingNet 后的残基特征 [L, D]
        self.part2_site_residue_features = {}  # 经过模块二 ActiveSiteEmbeddingNet 后的残基特征 [L, D]
        self.part2_probs = {}                  # 模块二基于距离计算出的预测概率 [L, C]


        self.train_dataset = EnzymeActiveSiteDataset3(
            csv_file=self.config['common']['train_data'],
            structure_path=self.config['common']['structure_path'],
            msa_dir='',                 # Deprecated arguments
            dca_dir='',                 # Deprecated arguments
            dssp_exec_path='',          # Deprecated arguments
            dssp_dir='',                # Deprecated arguments
            graphec_radius=0,           # Deprecated arguments
            protein_max_length=self.config['common']['max_seq_length'],
            max_msa_sequences=0,        # Deprecated arguments
            split='train',
            num_samples=None,
            esm_model_name='esm2_t33_650M_UR50D',
            msa_model_name='esm_msa1b_t12_100M_UR50S',
            num_workers=self.config['common']['num_workers'],
            preprocessing=None,         # Deprecated arguments
            preprocessing_dir=None,     # Deprecated arguments
            filter_incomplete_ec=True,
        )
        self.train_df = self.train_dataset.dataset_df

        self.valid_dataset = EnzymeActiveSiteDataset3(
            csv_file=self.config['common']['valid_data'],
            structure_path=self.config['common']['structure_path'],
            msa_dir='',                 # Deprecated arguments
            dca_dir='',                 # Deprecated arguments
            dssp_exec_path='',          # Deprecated arguments
            dssp_dir='',                # Deprecated arguments
            graphec_radius=0,           # Deprecated arguments
            protein_max_length=self.config['common']['max_seq_length'],
            max_msa_sequences=0,        # Deprecated arguments
            split='valid',
            num_samples=None,
            esm_model_name='esm2_t33_650M_UR50D',
            msa_model_name='esm_msa1b_t12_100M_UR50S',
            num_workers=self.config['common']['num_workers'],
            preprocessing=None,         # Deprecated arguments
            preprocessing_dir=None,     # Deprecated arguments
            filter_incomplete_ec=True,
        )
        self.valid_df = self.valid_dataset.dataset_df

        self.test_dataset = EnzymeActiveSiteDataset3(
            csv_file=self.config['common']['test_data'],
            structure_path=self.config['common']['structure_path'],
            msa_dir='',                 # Deprecated arguments
            dca_dir='',                 # Deprecated arguments
            dssp_exec_path='',          # Deprecated arguments
            dssp_dir='',                # Deprecated arguments
            graphec_radius=0,           # Deprecated arguments
            protein_max_length=self.config['common']['max_seq_length'],
            max_msa_sequences=0,        # Deprecated arguments
            split='test',
            num_samples=None,
            esm_model_name='esm2_t33_650M_UR50D',
            msa_model_name='esm_msa1b_t12_100M_UR50S',
            num_workers=self.config['common']['num_workers'],
            preprocessing=None,         # Deprecated arguments
            preprocessing_dir=None,     # Deprecated arguments
            filter_incomplete_ec=False,
        )
        print("test dataset not filtered for incomplete EC numbers")
        self.test_df = self.test_dataset.dataset_df

        # 检查并创建 Uniprot ID 到 sequence_identity 的映射
        self.uid_to_identity_map = {}
        if 'sequence_identity' in self.test_df.columns:
            print("Found 'sequence_identity' column in test data. Creating mapping...")
            # 确保 Uniprot ID 是字符串类型以进行匹配
            self.test_df['Uniprot ID'] = self.test_df['Uniprot ID'].astype(str)
            # 创建映射字典
            self.uid_to_identity_map = pd.Series(
                self.test_df.sequence_identity.values, 
                index=self.test_df['Uniprot ID']
            ).to_dict()
            print(f"Mapping created for {len(self.uid_to_identity_map)} samples.")
        else:
            print("Warning: 'sequence_identity' column not found in test data. Grouped analysis will be skipped.")

        self.entropy_threshold = float(self.config['common']['entropy_threshold'])


    def load_checkpoint(self):
        if self.args.part1_ckpt:
            print(f"[Tester] Loading ResidueEncoder Weights from {self.args.part1_ckpt}")
            ckpt = torch.load(self.args.part1_ckpt, map_location='cpu')
            state_dict = ckpt['model_state']['state_dict'] if 'model_state' in ckpt else ckpt
            self.model.residue_encoder.load_state_dict(state_dict, strict=True)
        if self.args.part2_site_ckpt:
            print(f"[Tester] Loading ActiveSiteEmbeddingNet from {self.args.part2_site_ckpt}")
            self.model.active_site_embedding_net.load_state_dict(torch.load(self.args.part2_site_ckpt, map_location='cpu'))
        self.model.to(self.device)



    def test(self):
        switch = self.config['common']['switch']

        self.inference_1(['train', 'valid', 'test'])
        self.evaluate_1(data_splits=['valid', 'test'])
        if switch == '2-ActiveSite':
            self.inference_2_site(data_splits=['train', 'valid', 'test'], input_source='part1')
            self.predict_hierarchical(feature_key='site')
        elif switch == '2-ActiveSite-EC4Only':
            self.inference_2_site(data_splits=['train', 'valid', 'test'], input_source='part1')
            self.predict_ec4_only(feature_key='site')
        elif switch == '2-ActiveSite-previous':
            self.inference_2_site(data_splits=['train', 'valid', 'test'], input_source='part1')
            self.evaluate_2_site(key='site')
        elif switch == '2-ActiveSite-hierarchical':
            # --- New Branch for True Hierarchical Inference ---
            print(">>> Starting 2-ActiveSite-hierarchical testing...")
            
            # 1. Run Inference with EC4 Model (Loaded by default)
            print("[Hierarchical] Step 1: Generating features using EC4 model...")
            self.inference_2_site(data_splits=['train', 'valid', 'test'], input_source='part1')
            # Deep copy to safe-guard features
            feats_ec4 = {
                'train': {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.part2_site_residue_features['train'].items()},
                'test': {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.part2_site_residue_features['test'].items()}
            }

            # 2. Determine EC3 Model Path & Load Weights
            ec4_ckpt_path = Path(self.args.part2_site_ckpt)
            # Logic: ../ActiveSiteEmbeddingNet/best.pt -> ../ActiveSiteEmbeddingNet_lvl3/best_model_site.pt
            ec3_ckpt_path = ec4_ckpt_path.parent.parent / 'ActiveSiteEmbeddingNet_lvl3' / 'best_model_site.pt'
            
            if not ec3_ckpt_path.exists():
                raise FileNotFoundError(f"EC3 checkpoint not found at {ec3_ckpt_path}. Cannot perform hierarchical inference.")
            
            print(f"[Hierarchical] Step 2: Loading EC3 model from {ec3_ckpt_path}...")
            # Load EC3 weights
            self.model.active_site_embedding_net.load_state_dict(torch.load(str(ec3_ckpt_path), map_location=self.device))
            
            # 3. Run Inference with EC3 Model
            print("[Hierarchical] Step 3: Generating features using EC3 model...")
            self.inference_2_site(data_splits=['train', 'valid', 'test'], input_source='part1')
            feats_ec3 = {
                'train': {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.part2_site_residue_features['train'].items()},
                'test': {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.part2_site_residue_features['test'].items()}
            }

            # Restore EC4 weights (Good practice)
            print(f"[Hierarchical] Restoring EC4 model weights from {ec4_ckpt_path}...")
            self.model.active_site_embedding_net.load_state_dict(torch.load(str(ec4_ckpt_path), map_location=self.device))

            # 4. Perform Hierarchical Prediction with Dual Features
            print("[Hierarchical] Step 4: Performing Split Evaluation...")
            self.predict_hierarchical_dual(feats_ec4, feats_ec3)

        else:
            print(f"Undefined switch for testing: {switch}")


    def predict_hierarchical_dual(self, feats_ec4: Dict, feats_ec3: Dict):
        """
        Hierarchical Inference using two different feature sets (EC4-trained and EC3-trained).
        
        Args:
            feats_ec4: Dict containing 'train' and 'test' residue features from EC4 model.
            feats_ec3: Dict containing 'train' and 'test' residue features from EC3 model.
        """
        # ... [Steps 1-4 remain unchanged: Building pools and evaluating] ...
        
        # -------------------------------------------------------------------------
        # Step 1: Build Reference Pools
        # -------------------------------------------------------------------------
        print("[Hierarchical-Dual] Building Reference Pool Level 4 (using EC4 Features)...")
        (_, _, _, _, ref_flat_4, ref_pools_4, _, _) = build_data_pools(
            dataframe=self.train_dataset.dataset_df,
            embedding_dict=feats_ec4['train'],
            num_workers=self.config['common']['num_workers'],
            ec_level=4, 
            cluster_mode="ec"
        )
        
        print("[Hierarchical-Dual] Building Reference Pool Level 3 (using EC3 Features)...")
        (_, _, _, _, ref_flat_3, ref_pools_3, _, _) = build_data_pools(
            dataframe=self.train_dataset.dataset_df,
            embedding_dict=feats_ec3['train'],
            num_workers=self.config['common']['num_workers'],
            ec_level=3,
            cluster_mode="ec"
        )

        # -------------------------------------------------------------------------
        # Step 2: Split Test Data by Entropy
        # -------------------------------------------------------------------------
        test_df = self.test_dataset.dataset_df.copy()
        
        if 'pred_ec4_top1(final)' not in test_df.columns or 'ec4_entropy_final' not in test_df.columns:
            raise ValueError("Test DataFrame missing 'pred_ec4_top1(final)' or 'ec4_entropy_final' columns.")

        # Split: Low Entropy -> Confident (Use Level 4)
        confident_mask = test_df['ec4_entropy_final'].astype(np.float64) < self.entropy_threshold
        
        df_confident = test_df[confident_mask].copy()
        df_uncertain = test_df[~confident_mask].copy() # High Entropy -> Uncertain (Fallback to Level 3)
        
        print(f"[Hierarchical-Dual] Split: {len(df_confident)} Confident (EC4 model), {len(df_uncertain)} Uncertain (EC3 model)")

        # -------------------------------------------------------------------------
        # Step 3: Special Processing (Assign Predictions)
        # -------------------------------------------------------------------------
        df_confident['valid_ec_list'] = df_confident['pred_ec4_top1(final)'].apply(lambda x: [str(x)])
        df_uncertain['valid_ec_list'] = df_uncertain['pred_ec4_top1(final)'].apply(lambda x: [str(x)])

        # -------------------------------------------------------------------------
        # Step 4: Build Query Pools using specific Features
        # -------------------------------------------------------------------------
        
        # 4a. Confident -> Use EC4 Features
        metrics_4, probs_4, labels_4 = {}, {}, {}
        if len(df_confident) > 0:
            print("[Hierarchical-Dual] Evaluating Confident Samples (Level 4)...")
            (query_id_4, _, _, labels_4_pool, query_flat_4, _, _, uid_off_4) = build_data_pools(
                dataframe=df_confident,
                embedding_dict=feats_ec4['test'],
                num_workers=self.config['common']['num_workers'],
                ec_level=4, 
                cluster_mode="ec"
            )
            metrics_4, probs_4 = evaluate_site_type_prediction(
                reference_residue_features=ref_flat_4,
                query_residue_features=query_flat_4,
                ec_site_pools_indexed_reference=ref_pools_4,
                id_ec_query=query_id_4,
                labels_dict_query=labels_4_pool,
                uid_to_offset_query=uid_off_4,
                device=self.device,
                prob_mode="distance"
            )
            labels_4 = labels_4_pool

        # 4b. Uncertain -> Use EC3 Features
        metrics_3, probs_3, labels_3 = {}, {}, {}
        if len(df_uncertain) > 0:
            print("[Hierarchical-Dual] Evaluating Uncertain Samples (Level 3)...")
            (query_id_3, _, _, labels_3_pool, query_flat_3, _, _, uid_off_3) = build_data_pools(
                dataframe=df_uncertain,
                embedding_dict=feats_ec3['test'],
                num_workers=self.config['common']['num_workers'],
                ec_level=3, 
                cluster_mode="ec"
            )
            metrics_3, probs_3 = evaluate_site_type_prediction(
                reference_residue_features=ref_flat_3,
                query_residue_features=query_flat_3,
                ec_site_pools_indexed_reference=ref_pools_3,
                id_ec_query=query_id_3,
                labels_dict_query=labels_3_pool,
                uid_to_offset_query=uid_off_3,
                device=self.device,
                prob_mode="distance"
            )
            labels_3 = labels_3_pool

        # -------------------------------------------------------------------------
        # Step 5: Merge Results and Prepare for Reporting (FIXED)
        # -------------------------------------------------------------------------
        merged_probs = {**probs_4, **probs_3}
        self.part2_probs['test'] = merged_probs

        # --- 正确的合并逻辑 ---
        # 1. 将 metrics 字典列表转换为 uid-keyed 字典
        uid_to_metrics = defaultdict(dict)
        sorted_uids_4 = sorted(probs_4.keys())
        sorted_uids_3 = sorted(probs_3.keys())

        for i, uid in enumerate(sorted_uids_4):
            for metric, values in metrics_4.items():
                if i < len(values): uid_to_metrics[uid][metric] = values[i]
        
        for i, uid in enumerate(sorted_uids_3):
            for metric, values in metrics_3.items():
                if i < len(values): uid_to_metrics[uid][metric] = values[i]

        # 2. 从 uid_to_metrics 构建 DataFrame
        df_results = pd.DataFrame.from_dict(uid_to_metrics, orient='index')
        df_results.reset_index(inplace=True)
        df_results.rename(columns={'index': 'Uniprot ID'}, inplace=True)
        
        # 3. 添加 sequence_identity 信息
        if self.uid_to_identity_map:
            df_results['sequence_identity'] = df_results['Uniprot ID'].map(self.uid_to_identity_map).fillna('unknown')
        else:
            df_results['sequence_identity'] = 'unknown'
        # --- 结束修改 ---

        # -------------------------------------------------------------------------
        # Step 6: 分组报告和整体报告 (UPDATED)
        # -------------------------------------------------------------------------
        
        # 6a. Hierarchical Split Report (Confident vs Uncertain)
        # ... (这部分代码保持不变) ...

        # 6b. 整体指标报告 (Overall)
        overall_metrics = df_results.select_dtypes(include=np.number).mean().to_dict()
        print(f"\n[Hierarchical-Dual] OVERALL metrics (Averaged across {len(df_results)} samples):")
        for metric_name, value in sorted(overall_metrics.items()):
            print(f"  {metric_name.replace('-', '_')}: {value:.4f}")

        # --- 修改：按 Sequence Identity 分组报告 (打印所有指标) ---
        if self.uid_to_identity_map and 'unknown' not in df_results['sequence_identity'].unique():
            print("\n" + "="*80)
            print(" " * 25 + "Performance by Sequence Identity Group")
            print("="*80)
            
            # 动态获取所有可用的指标列
            all_metric_cols = sorted([col for col in df_results.columns if col not in ['Uniprot ID', 'sequence_identity']])

            # 使用 groupby().agg() 一次性计算所有指标的均值
            grouped_stats = df_results.groupby('sequence_identity')[all_metric_cols].agg(['mean', 'count'])
            
            # 为了更好的可读性，我们可以转置并打印
            for group_name, group_data in grouped_stats.iterrows():
                count = int(group_data.xs('count', level=1).iloc[0]) # 获取该组的样本数
                print(f"--- Group: {group_name} (Count: {count}) ---")
                
                # 只打印均值
                mean_stats = group_data.xs('mean', level=1)
                for metric, value in mean_stats.items():
                    print(f"  {metric:<25}: {value:.4f}")
                print("-" * 40)
            print("="*80 + "\n")
        # --- 结束修改 ---
        
        # 返回合并后的指标字典 (方便外部调用)
        merged_metrics = df_results.drop(columns=['Uniprot ID', 'sequence_identity']).to_dict(orient='list')
        
        return merged_metrics

    # def inference_1(self, data_splits: List[str]):
    #     """
    #     运行模块一推理，缓存 features和 Logits。
    #     缓存内容: 
    #         features_cache[data_split]['part1'] = {uid: tensor}, data_split ∈ {train, valid, test}
    #         probs_cache[data_split]['part1'] = {uid: tensor}, data_split ∈ {train, valid, test}
    #     """
    #     print(f"[Tester] Running Part 1 Inference on {data_splits} set...")

    #     for data_split in data_splits:
    #         print(f"Processing {data_split} data...")
    #         if data_split == 'train':
    #             dataset = self.train_dataset
    #         elif data_split == 'valid':
    #             dataset = self.valid_dataset
    #         elif data_split == 'test':
    #             dataset = self.test_dataset
    #         else:
    #             raise ValueError(f"Unknown split: {data_split}")

    #         dataloader = DataLoader(
    #             dataset,
    #             batch_size=24,
    #             shuffle=False,
    #             num_workers=self.config['common']['num_workers'],
    #             collate_fn=collate_enzyme_batch,
    #             pin_memory=True
    #         )

    #         residue_features_dict = {}
    #         protein_features_dict = {}
    #         probs_dict = {}
    #         with torch.no_grad():
    #             for batch in tqdm(dataloader, desc=f"Extracting Features for {data_split} set"):
    #                 if batch is None: continue
    #                 batch = move_batch_to_device(batch, self.device)
    #                 with autocast(enabled=self.use_amp, dtype=self.dtype):
    #                     padded_feats, protein_mask = self.model.residue_encoder(batch, return_features=True)
    #                     flat_feats = padded_feats[protein_mask.bool()]
    #                     logits = self.model.residue_encoder.prediction_head(flat_feats)
    #                     probs = torch.softmax(logits, dim=-1)
    #                 num_nodes_per_graph = batch['protein_graph'].num_residues.tolist()
    #                 residue_features_list = torch.split(flat_feats, num_nodes_per_graph)
    #                 residue_probs_list = torch.split(probs, num_nodes_per_graph)
    #                 for i, uniprot_id in enumerate(batch['uniprot_ids']):
    #                     residues_tensor = residue_features_list[i].cpu()    # [L, D]
    #                     assert residues_tensor.size(0) == num_nodes_per_graph[i], f"残基数量不匹配: {residues_tensor.size(0)} vs {num_nodes_per_graph[i]} for protein {uniprot_id}"
    #                     residue_features_dict[uniprot_id] = residues_tensor.clone()
    #                     pooled_embedding = residues_tensor.mean(dim=0)
    #                     protein_features_dict[uniprot_id] = pooled_embedding.to(self.dtype)
    #                     if data_split == 'test' or data_split == 'valid':
    #                         probs_dict[uniprot_id] = residue_probs_list[i].cpu().float()

    #         self.part1_residue_features[data_split] = residue_features_dict
    #         self.part1_protein_features[data_split] = protein_features_dict
    #         self.part1_probs[data_split] = probs_dict
    #         print(f"Finished Inference on Part 1 for {data_split} data. {len(residue_features_dict)} samples processed.")
        

    def inference_1(self, data_splits: List[str]):
        """
        运行模块一推理，缓存 features和 Logits。
        缓存文件名格式: stage_1_cache_{ModelID}_{DatasetID}_{Split}.pt
        例如: stage_1_cache_ResidueEncoder_checkpoint_epoch_31_SwissProt_Enzyme_CDHIT60_train.pt
        """
        print(f"[Tester] Running Part 1 Inference on {data_splits} set...")

        # 1. 确定模型标识 (Model Identifier)
        # 逻辑: 提取 part1_ckpt 的 父目录名(ResidueEncoder) 和 文件主名(checkpoint_epoch_31) 进行拼接
        model_id = "UnknownModel"
        if self.args.part1_ckpt:
            p = Path(self.args.part1_ckpt)
            model_id = f"{p.parent.name}_{p.stem}" # e.g. ResidueEncoder_checkpoint_epoch_31
        
        # 2. 确定数据集标识 (Dataset Identifier) - 固定为 SwissProt_Enzyme_CDHIT60
        dataset_id = "SwissProt_Enzyme_CDHIT60"
        
        # 3. 确定缓存基础目录 (与训练数据同目录)
        train_data_path = self.config['common']['train_data']
        cache_base_dir = os.path.dirname(train_data_path)

        for data_split in data_splits:
            print(f"Processing {data_split} data...")
            
            # 选择对应的数据集对象
            if data_split == 'train':
                dataset = self.train_dataset
            elif data_split == 'valid':
                dataset = self.valid_dataset
            elif data_split == 'test':
                dataset = self.test_dataset
            else:
                raise ValueError(f"Unknown split: {data_split}")

            # 4. 构建完整的缓存文件名
            # 格式: stage_1_cache_ResidueEncoder_checkpoint_epoch_31_SwissProt_Enzyme_CDHIT60_train.pt
            cache_filename = f"stage_1_cache_{model_id}_{dataset_id}_{data_split}.pt"
            cache_path = os.path.join(cache_base_dir, cache_filename)

            # 5. 检查缓存是否存在
            if os.path.exists(cache_path):
                print(f"[Tester] Found cache for {data_split}: {cache_path}")
                try:
                    print(f"[Tester] Loading from cache...")
                    cache_data = torch.load(cache_path, map_location='cpu')
                    self.part1_residue_features[data_split] = cache_data['residue_features']
                    self.part1_protein_features[data_split] = cache_data['protein_features']
                    self.part1_probs[data_split] = cache_data['probs']
                    print(f"[Tester] Successfully loaded {len(self.part1_residue_features[data_split])} samples from cache.")
                    continue # 跳过后续的推理循环，直接处理下一个 split
                except Exception as e:
                    print(f"[Tester] Failed to load cache: {e}. Re-running inference.")

            # 6. 运行推理 (如果没有缓存或加载失败)
            dataloader = DataLoader(
                dataset,
                batch_size=24,
                shuffle=False,
                num_workers=self.config['common']['num_workers'],
                collate_fn=collate_enzyme_batch,
                pin_memory=True
            )

            residue_features_dict = {}
            protein_features_dict = {}
            probs_dict = {}
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Extracting Features for {data_split} set"):
                    if batch is None: continue
                    batch = move_batch_to_device(batch, self.device)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        padded_feats, protein_mask = self.model.residue_encoder(batch, return_features=True)
                        flat_feats = padded_feats[protein_mask.bool()]
                        logits = self.model.residue_encoder.prediction_head(flat_feats)
                        probs = torch.softmax(logits, dim=-1)
                    num_nodes_per_graph = batch['protein_graph'].num_residues.tolist()
                    residue_features_list = torch.split(flat_feats, num_nodes_per_graph)
                    residue_probs_list = torch.split(probs, num_nodes_per_graph)
                    for i, uniprot_id in enumerate(batch['uniprot_ids']):
                        residues_tensor = residue_features_list[i].cpu()    # [L, D]
                        assert residues_tensor.size(0) == num_nodes_per_graph[i], f"Residue count mismatch: {residues_tensor.size(0)} vs {num_nodes_per_graph[i]} for {uniprot_id}"
                        residue_features_dict[uniprot_id] = residues_tensor.clone()
                        pooled_embedding = residues_tensor.mean(dim=0)
                        protein_features_dict[uniprot_id] = pooled_embedding.to(self.dtype)
                        if data_split == 'test' or data_split == 'valid':
                            probs_dict[uniprot_id] = residue_probs_list[i].cpu().float()

            # 7. 保存结果到缓存文件
            print(f"[Tester] Saving results to cache: {cache_path}")
            cache_data = {
                'residue_features': residue_features_dict,
                'protein_features': protein_features_dict,
                'probs': probs_dict
            }
            try:
                torch.save(cache_data, cache_path)
                print(f"[Tester] Cache saved.")
            except Exception as e:
                print(f"[Tester] Warning: Failed to save cache: {e}")

            self.part1_residue_features[data_split] = residue_features_dict
            self.part1_protein_features[data_split] = protein_features_dict
            self.part1_probs[data_split] = probs_dict
            print(f"Finished Inference on Part 1 for {data_split} data. {len(residue_features_dict)} samples processed.")



    def inference_2_site(self, data_splits: List[str], input_source: str):
        """
        运行 ActiveSiteEmbeddingNet 推理。
        Args:
            input_source: 'part1' (来自模块一) 或 'ec' (来自 ECEmbeddingNet)
        """
        print(f"[Tester] Running ActiveSiteEmbeddingNet Inference on {data_splits} set (Input: {input_source})...")
        
        for data_split in data_splits:
            if input_source == 'part1':
                source_dict = self.part1_residue_features[data_split]
            elif input_source == 'ec':
                source_dict = self.part2_ec_residue_features[data_split]
            else:
                raise ValueError(f"Invalid input_source: {input_source}")
            if source_dict is None:   raise ValueError(f"Input residue features for ActiveSiteEmbeddingNet '{data_split}' from '{input_source}' not found.")
            dataset = ResidueFeatureDataset(source_dict)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=self.config['common']['num_workers'], collate_fn=collate_residue_features, pin_memory=True)

            site_aware_residue_dict = {}
            with torch.no_grad():
                for uniprot_ids, concatenated_residues, lengths in tqdm(dataloader, desc=f"Inferring Site-aware features for {data_split}"):
                    concatenated_residues = concatenated_residues.to(self.device, non_blocking=True)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        concatenated_embs = self.model.active_site_embedding_net(concatenated_residues)
                    split_embs = torch.split(concatenated_embs, lengths)
                    for uniprot_id, emb_tensor in zip(uniprot_ids, split_embs):
                        site_aware_residue_dict[uniprot_id] = emb_tensor.cpu()

            for uniprot_id, res_data in source_dict.items():
                if isinstance(res_data, list) and not res_data:
                    site_aware_residue_dict[uniprot_id] = []

            self.part2_site_residue_features[data_split] = site_aware_residue_dict
            print(f"Finished Inference on ActiveSiteEmbeddingNet for {data_split} data.")



    def inference_2_logits(self, feature_key: str) -> Dict[str, torch.Tensor]:
        """
        使用 evaluate_site_type_prediction 计算基于距离的概率，并全局缓存。
        
        Args:
            feature_key: 使用哪种特征计算概率 ('ec' 或 'site')
        Returns:
            query_labels: 测试集的真实标签（用于 Stacking Dataset 构建）
        """
        print(f"[Tester] Generating Part 2 probabilities using '{feature_key}' features...")

        if feature_key == 'ec':
            ref_feats = self.part2_ec_residue_features['train']
            query_feats = self.part2_ec_residue_features['test']
        elif feature_key == 'site':
            ref_feats = self.part2_site_residue_features['train']
            query_feats = self.part2_site_residue_features['test']
        else:
            raise ValueError(f"Invalid feature_key: {feature_key}")

        (_, _, _, _, ref_flat, ref_ec_pools, _, _) = build_data_pools(dataframe=self.train_df, embedding_dict=ref_feats, num_workers=self.config['common']['num_workers'], ec_level=self.config['common']['ec_level'])
        (query_id_ec, _, _, query_labels, query_flat, _, _, query_uid_offset) = build_data_pools(dataframe=self.test_df, embedding_dict=query_feats, num_workers=self.config['common']['num_workers'], ec_level=self.config['common']['ec_level'])

        _, per_sample_probs = evaluate_site_type_prediction(
            reference_residue_features=ref_flat,
            query_residue_features=query_flat,
            ec_site_pools_indexed_reference=ref_ec_pools,
            id_ec_query=query_id_ec,
            labels_dict_query=query_labels,
            uid_to_offset_query=query_uid_offset,
            device=self.device,
            prob_mode="distance"
        )
        
        self.part2_probs['test'] = per_sample_probs
        return query_labels



    def evaluate_1(self, data_splits: List[str] = ['test']):
        """
        评估指定数据集的预测结果。
        
        Args:
            data_splits: 要评估的数据集列表，例如 ['valid', 'test']
        """
        num_classes = self.config['model']['ResidueEncoderGamma']['num_class']
        
        for split in data_splits:
            print(f"Evaluating Part 1 predictions on {split.upper()} set")
            if split == 'valid':
                probs_dict = self.part1_probs['valid']
                feats = self.part1_residue_features['valid']
                dataframe = self.valid_df
            elif split == 'test':
                probs_dict = self.part1_probs['test']
                feats = self.part1_residue_features['test']
                dataframe = self.test_df
            else:
                raise ValueError(f"Unknown split: {split}")
            
            (_, _, _, labels_dict, _, _, _, _) = build_data_pools(
                dataframe=dataframe, 
                embedding_dict=feats, 
                num_workers=self.config['common']['num_workers'], 
                ec_level=self.config['common']['ec_level']
            )
            

            all_metrics = defaultdict(list)
            sorted_uids = sorted(list(set(probs_dict.keys()) & set(labels_dict.keys())))
            print(f"Number of samples to evaluate: {len(sorted_uids)}")
            for uid in sorted_uids:
                # 获取单个样本的概率和标签 [seq_len] 或 [seq_len, num_classes]
                sample_probs = probs_dict[uid]
                sample_labels = labels_dict[uid]
                
                if isinstance(sample_probs, torch.Tensor):
                    sample_probs = sample_probs.cpu()
                if isinstance(sample_labels, torch.Tensor):
                    sample_labels = sample_labels.cpu()
                
                sample_preds = torch.argmax(sample_probs, dim=-1)  # [seq_len]
                assert sample_preds.shape == sample_labels.shape, f"Shape mismatch for {uid}: preds {sample_preds.shape}, labels {sample_labels.shape}"
                
                # 添加 batch 维度以兼容 compute_*_classification_metrics 函数
                preds = sample_preds.unsqueeze(0)  # [1, seq_len]
                labels = sample_labels.unsqueeze(0)  # [1, seq_len]
                probs = sample_probs.unsqueeze(0)  # [1, seq_len, num_classes]
                valid_mask = torch.ones_like(labels).bool()  # [1, seq_len]
                
                if num_classes == 2:
                    # 二分类
                    prob_for_metrics = probs[:, :, 1]
                    batch_metrics = compute_binary_classification_metrics(
                        predictions=preds,
                        probabilities=prob_for_metrics,
                        labels=labels,
                        valid_mask=valid_mask,
                        metrics=['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc', 'ap', 'mcc']
                    )
                    for metric_name, metric_values in batch_metrics.items():
                        all_metrics[metric_name].extend(metric_values)
                else:
                    # 计算二分类指标（活性位点vs非活性位点），计算方式与EasIFA一致
                    binary_labels = (labels != 0).float()
                    has_positive = (binary_labels * valid_mask.float()).sum() > 0
                    
                    if has_positive:
                        binary_preds = (preds != 0).float()
                        binary_probs = 1 - probs[:, :, 0]  # 非背景类的总概率
                        
                        binary_metrics = compute_binary_classification_metrics(
                            predictions=binary_preds,
                            probabilities=binary_probs,
                            labels=binary_labels,
                            valid_mask=valid_mask,
                            metrics=['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc', 'ap', 'mcc']
                        )
                        
                        for metric_name, metric_values in binary_metrics.items():
                            all_metrics[f"binary_{metric_name}"].extend(metric_values)
                    
                    # 计算多分类指标，计算方式与EasIFA一致
                    multi_metrics = compute_multiple_classification_metrics(
                        pred=preds,
                        gt=labels,
                        num_site_types=num_classes,
                        valid_mask=valid_mask
                    )
                    for metric_name, values in multi_metrics.items():
                        if values:
                            metric_key = metric_name.replace('-', '_')
                            all_metrics[metric_key].extend(values)
            
            # 计算所有指标的平均值
            if all_metrics:
                classification_metrics = {}
                for metric_name, values in all_metrics.items():
                    if values:
                        classification_metrics[metric_name] = sum(values) / len(values)
                
                print(f"\nOverall metrics for {split.upper()} set (averaged across {len(sorted_uids)} samples):")
                for metric_name, value in sorted(classification_metrics.items()):
                    print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"\nNo valid metrics computed for {split.upper()} set")



    def evaluate_2_site(self, key: str):
        """
        Args:
            key: 'ec' 或 'site'
        """
        if key == 'ec':
            ref_feats = self.part2_ec_residue_features['train']
            query_feats = self.part2_ec_residue_features['test']
        elif key == 'site':
            ref_feats = self.part2_site_residue_features['train']
            query_feats = self.part2_site_residue_features['test']
        else:
             raise ValueError(f"Invalid feature_key: {key}")

        (_, _, _, _, ref_flat, ref_ec_pools, _, _) = build_data_pools(dataframe=self.train_df, embedding_dict=ref_feats, num_workers=self.config['common']['num_workers'], ec_level=self.config['common']['ec_level'])
        (query_id_ec, _, _, query_labels, query_flat, _, _, query_uid_offset) = build_data_pools(dataframe=self.test_df, embedding_dict=query_feats, num_workers=self.config['common']['num_workers'], ec_level=self.config['common']['ec_level'])
        
        all_metrics, per_sample_probs = evaluate_site_type_prediction(
            reference_residue_features=ref_flat,
            query_residue_features=query_flat,
            ec_site_pools_indexed_reference=ref_ec_pools,
            id_ec_query=query_id_ec,
            labels_dict_query=query_labels,
            uid_to_offset_query=query_uid_offset,
            device=self.device,
            prob_mode="distance"
        )

        # --- New Logic for Saving Results ---
        print(f"\n[Tester] Saving detailed results for {key}...")
        
        # 1. Prepare Output Directory
        output_dir = self.config.get('test', {}).get('output_dir', 'results')
        if not output_dir: output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)

        results_list = []
        # Ensure we iterate in a deterministic order consistent with how per_sample_probs was built
        evaluated_uids = sorted(per_sample_probs.keys())
        
        for i, uid in enumerate(evaluated_uids):
            row_data = {}
            row_data['Uniprot ID'] = str(uid)
            
            # Get predictions
            probs = per_sample_probs[uid]     # [L, C]
            preds = torch.argmax(probs, dim=-1) # [L]
            
            # Format prediction strings
            pred_labels_str, pred_types_str = format_prediction_to_string(preds)
            row_data['clustersite_predicted_site_labels'] = pred_labels_str
            row_data['clustersite_predicted_site_types'] = pred_types_str
            
            # Add metrics (aligned by index i because all_metrics are constructed from sorted UIDs)
            for metric_name, values_list in all_metrics.items():
                if i < len(values_list):
                    row_data[metric_name] = values_list[i]
            
            results_list.append(row_data)

        # 2. Merge and Save
        if results_list:
            df_preds = pd.DataFrame(results_list)
            
            # Use a copy of test_df to merge
            test_df_copy = self.test_df.copy()
            test_df_copy['Uniprot ID'] = test_df_copy['Uniprot ID'].astype(str)
            
            # Left join to map predictions back to original sample info
            df_final = pd.merge(test_df_copy, df_preds, on='Uniprot ID', how='left')
            
            # Save CSV
            csv_path = os.path.join(output_dir, f"clustersite_{key}_test_results.csv")
            df_final.to_csv(csv_path, index=False)
            print(f"Detailed results saved to {csv_path}")
            
            # 3. Compute and Save Summary Metrics
            summary_metrics = {}
            # Columns to exclude from summary
            exclude_cols = ['Uniprot ID', 'clustersite_predicted_site_labels', 'clustersite_predicted_site_types']
            target_metric_cols = [c for c in df_preds.columns if c not in exclude_cols]
            
            print("\n" + "="*50)
            print(f"Final Test Results ({key}) - Average over valid predictions:")
            
            valid_count = df_final['clustersite_predicted_site_labels'].notna().sum()
            print(f"Total CSV Rows: {len(df_final)}")
            print(f"Valid Predictions: {valid_count}")

            for col in target_metric_cols:
                if col in df_final.columns and pd.api.types.is_numeric_dtype(df_final[col]):
                    mean_val = df_final[col].mean()
                    summary_metrics[col] = float(mean_val)
                    print(f"{col}: {mean_val:.4f}")
            
            print("="*50 + "\n")

            # Save JSON
            json_path = os.path.join(output_dir, f"clustersite_{key}_metrics_summary.json")
            with open(json_path, 'w') as f:
                json.dump(summary_metrics, f, indent=4)
            print(f"Metrics summary saved to {json_path}")
        else:
            print("Warning: No results collected to save.")

        # Print overall metrics
        if all_metrics:
            final_metrics = {}
            print(f"\nevaluate_2_site, {key}: Overall metrics (averaged across all samples):")
            for metric_name, values in all_metrics.items():
                if values:
                    metric_key = metric_name.replace('-', '_')
                    final_metrics[metric_key] = sum(values) / len(values)
            
            for metric_name, value in sorted(final_metrics.items()):
                print(f"  {metric_name}: {value:.4f}")



    def predict_hierarchical(self, feature_key: str):
        """
        Core function for Uncertainty-Aware Hierarchical Inference.
        Merges evaluation, logic splitting, and reporting.
        """
        # Select Features
        if feature_key == 'ec':
            ref_feats = self.part2_ec_residue_features['train']
            query_feats = self.part2_ec_residue_features['test']
        elif feature_key == 'site':
            ref_feats = self.part2_site_residue_features['train']
            query_feats = self.part2_site_residue_features['test']
        else:
            raise ValueError(f"Invalid feature_key: {feature_key}")

        # -------------------------------------------------------------------------
        # Step 1: Build Reference Pools (Training Data)
        # -------------------------------------------------------------------------
        print("[Hierarchical] Building Reference Pool Level 4 (for Confident samples)...")
        (_, _, _, _, ref_flat_4, ref_pools_4, _, _) = build_data_pools(
            dataframe=self.train_dataset.dataset_df,
            embedding_dict=ref_feats,
            num_workers=self.config['common']['num_workers'],
            ec_level=4, 
            cluster_mode="ec"
        )
        
        print("[Hierarchical] Building Reference Pool Level 3 (for Uncertain samples)...")
        (_, _, _, _, ref_flat_3, ref_pools_3, _, _) = build_data_pools(
            dataframe=self.train_dataset.dataset_df,
            embedding_dict=ref_feats,
            num_workers=self.config['common']['num_workers'],
            ec_level=3,
            cluster_mode="ec"
        )

        # -------------------------------------------------------------------------
        # Step 2: Split Test Data by Entropy
        # -------------------------------------------------------------------------
        test_df = self.test_dataset.dataset_df.copy()
        
        if 'pred_ec4_top1(final)' not in test_df.columns or 'ec4_entropy_final' not in test_df.columns:
            raise ValueError("Test DataFrame missing 'pred_ec4_top1(final)' or 'ec4_entropy_final' columns.")

        # Split: Low Entropy -> Confident (Use Level 4)
        confident_mask = test_df['ec4_entropy_final'].astype(np.float64) < self.entropy_threshold
        
        df_confident = test_df[confident_mask].copy()
        df_uncertain = test_df[~confident_mask].copy() # High Entropy -> Uncertain (Fallback to Level 3)
        
        print(f"[Hierarchical] Test Data Split: {len(df_confident)} Confident (EC4), {len(df_uncertain)} Uncertain (EC3-Fallback)")

        # -------------------------------------------------------------------------
        # Step 3: Special Processing (Assign Predictions)
        # -------------------------------------------------------------------------
        # Assign 'pred_ec4_top1(final)' to 'valid_ec_list'.
        # Since we updated process_row to handle strings, we can just assign the string column directly 
        # (or wrap in list if we want to be explicit, but process_row handles single strings now).
        # We will wrap in list to be safe and consistent with previous turn's logic.
        
        # Confident: Level 4
        df_confident['valid_ec_list'] = df_confident['pred_ec4_top1(final)'].apply(lambda x: [str(x)])
        
        # Uncertain: Level 4 Prediction (Will be truncated to Level 3 by build_data_pools with ec_level=3)
        df_uncertain['valid_ec_list'] = df_uncertain['pred_ec4_top1(final)'].apply(lambda x: [str(x)])

        # -------------------------------------------------------------------------
        # Step 4: Build Query Pools
        # -------------------------------------------------------------------------
        print("[Hierarchical] Building Query Pool Level 4...")
        (query_id_4, _, _, labels_4, query_flat_4, _, _, uid_off_4) = build_data_pools(
            dataframe=df_confident,
            embedding_dict=query_feats,
            num_workers=self.config['common']['num_workers'],
            ec_level=4, 
            cluster_mode="ec"
        )
        
        print("[Hierarchical] Building Query Pool Level 3...")
        (query_id_3, _, _, labels_3, query_flat_3, _, _, uid_off_3) = build_data_pools(
            dataframe=df_uncertain,
            embedding_dict=query_feats,
            num_workers=self.config['common']['num_workers'],
            ec_level=3, 
            cluster_mode="ec"
        )

        # -------------------------------------------------------------------------
        # Step 5: Evaluate Separately
        # -------------------------------------------------------------------------
        metrics_4, probs_4 = {}, {}
        if len(df_confident) > 0:
            metrics_4, probs_4 = evaluate_site_type_prediction(
                reference_residue_features=ref_flat_4,
                query_residue_features=query_flat_4,
                ec_site_pools_indexed_reference=ref_pools_4,
                id_ec_query=query_id_4,
                labels_dict_query=labels_4,
                uid_to_offset_query=uid_off_4,
                device=self.device,
                prob_mode="distance"
            )
        
        metrics_3, probs_3 = {}, {}
        if len(df_uncertain) > 0:
            metrics_3, probs_3 = evaluate_site_type_prediction(
                reference_residue_features=ref_flat_3,
                query_residue_features=query_flat_3,
                ec_site_pools_indexed_reference=ref_pools_3,
                id_ec_query=query_id_3,
                labels_dict_query=labels_3,
                uid_to_offset_query=uid_off_3,
                device=self.device,
                prob_mode="distance"
            )

        # -------------------------------------------------------------------------
        # Step 6: Merge Results
        # -------------------------------------------------------------------------
        merged_metrics = defaultdict(list)
        for k, v in metrics_4.items(): merged_metrics[k].extend(v)
        for k, v in metrics_3.items(): merged_metrics[k].extend(v)
        
        merged_probs = {**probs_4, **probs_3}
        merged_labels = {**labels_4, **labels_3}

        # -------------------------------------------------------------------------
        # Step 7: Print Split Report (Merged from _print_split_report)
        # -------------------------------------------------------------------------
        print(f"\n[Hierarchical Report] Entropy Threshold = {self.entropy_threshold}")
        
        def get_avg_mcc(m):
            if not m or 'multi-class mcc' not in m: return 0.0
            return sum(m['multi-class mcc']) / len(m['multi-class mcc'])
            
        mcc_4 = get_avg_mcc(metrics_4)
        mcc_3 = get_avg_mcc(metrics_3)
        n_4 = len(next(iter(metrics_4.values()))) if metrics_4 else 0
        n_3 = len(next(iter(metrics_3.values()))) if metrics_3 else 0
        
        print(f"  > Confident Group (Level 4): N={n_4}, MCC={mcc_4:.4f}")
        print(f"  > Uncertain Group (Level 3): N={n_3}, MCC={mcc_3:.4f}")

        # -------------------------------------------------------------------------
        # Step 8: Print Overall Metrics (Merged from evaluate_2_site)
        # -------------------------------------------------------------------------
        if merged_metrics:
            final_metrics = {}
            print(f"\n[{feature_key}] OVERALL metrics (Averaged across {len(next(iter(merged_metrics.values())))} samples):")
            for metric_name, values in merged_metrics.items():
                if values:
                    metric_key = metric_name.replace('-', '_')
                    final_metrics[metric_key] = sum(values) / len(values)
            
            for metric_name, value in sorted(final_metrics.items()):
                print(f"  {metric_name}: {value:.4f}")
        
        return merged_metrics, merged_probs, merged_labels
    


    def predict_ec4_only(self, feature_key: str):
        """
        Force using EC Level 4 prototypes for ALL test samples, ignoring entropy/uncertainty.
        Uses 'pred_ec4_top1(final)' as the target cluster.
        """
        # Select Features
        if feature_key == 'ec':
            ref_feats = self.part2_ec_residue_features['train']
            query_feats = self.part2_ec_residue_features['test']
        elif feature_key == 'site':
            ref_feats = self.part2_site_residue_features['train']
            query_feats = self.part2_site_residue_features['test']
        else:
            raise ValueError(f"Invalid feature_key: {feature_key}")

        # -------------------------------------------------------------------------
        # Step 1: Build Reference Pools (Training Data) - Level 4 Only
        # -------------------------------------------------------------------------
        print("[EC4-Only] Building Reference Pool Level 4...")
        (_, _, _, _, ref_flat, ref_pools, _, _) = build_data_pools(
            dataframe=self.train_dataset.dataset_df,
            embedding_dict=ref_feats,
            num_workers=self.config['common']['num_workers'],
            ec_level=4, 
            cluster_mode="ec"
        )

        # -------------------------------------------------------------------------
        # Step 2: Prepare Test Data
        # -------------------------------------------------------------------------
        test_df = self.test_dataset.dataset_df.copy()
        
        if 'pred_ec4_top1(final)' not in test_df.columns:
            raise ValueError("Test DataFrame missing 'pred_ec4_top1(final)' column.")

        # Assign predictions to valid_ec_list for all samples
        # Wrap in list to satisfy process_row requirements
        test_df['valid_ec_list'] = test_df['pred_ec4_top1(final)'].apply(lambda x: [str(x)])
        
        print(f"[EC4-Only] Test Data Size: {len(test_df)} samples")

        # -------------------------------------------------------------------------
        # Step 3: Build Query Pools - Level 4 Only
        # -------------------------------------------------------------------------
        print("[EC4-Only] Building Query Pool Level 4...")
        (query_id, _, _, labels, query_flat, _, _, uid_off) = build_data_pools(
            dataframe=test_df,
            embedding_dict=query_feats,
            num_workers=self.config['common']['num_workers'],
            ec_level=4, 
            cluster_mode="ec"
        )

        # -------------------------------------------------------------------------
        # Step 4: Evaluate
        # -------------------------------------------------------------------------
        all_metrics, per_sample_probs = evaluate_site_type_prediction(
            reference_residue_features=ref_flat,
            query_residue_features=query_flat,
            ec_site_pools_indexed_reference=ref_pools,
            id_ec_query=query_id,
            labels_dict_query=labels,
            uid_to_offset_query=uid_off,
            device=self.device,
            prob_mode="distance"
        )

        # -------------------------------------------------------------------------
        # Step 5: Report Results
        # -------------------------------------------------------------------------
        if all_metrics:
            final_metrics = {}
            print(f"\n[{feature_key}] EC4-ONLY metrics (Averaged across {len(next(iter(all_metrics.values())))} samples):")
            for metric_name, values in all_metrics.items():
                if values:
                    metric_key = metric_name.replace('-', '_')
                    final_metrics[metric_key] = sum(values) / len(values)
            
            for metric_name, value in sorted(final_metrics.items()):
                print(f"  {metric_name}: {value:.4f}")
        
        # Update shared probs for Stacking (if needed later)
        self.part2_probs['test'] = per_sample_probs
        
        return all_metrics, per_sample_probs, labels

 




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EvoSite Inference Script")
    
    parser.add_argument("--config", type=str, required=True, help="Required, Path to inference configuration file")
    parser.add_argument("--tester", type=str, default="clustersite", help="Required, Type of tester to use")
    parser.add_argument("--part1_ckpt", type=str, help="Path to ResidueEncoderBeta checkpoint")
    parser.add_argument("--part2_ec_ckpt", type=str, help="Path to ECEmbeddingNet checkpoint")
    parser.add_argument("--part2_site_ckpt", type=str, help="Path to ActiveSiteEmbeddingNet checkpoint")
    parser.add_argument("--stacking_ckpt", type=str, help="Path to StackingC checkpoint")


    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Optional, Path to trained model checkpoint"
    )

    # Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional, Override output directory from config"
    )
    
    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional, Override batch size from config"
    )
    
    # Test data
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Optional, Override test data path from config"
    )
    
    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Optional, Enable debug mode with additional logging"
    )
    
    return parser.parse_args()



def main():
    """Main inference function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config parameters from command line
    if args.output_dir:
        config['test']['output_dir'] = args.output_dir
    if args.batch_size:
        config['test']['batch_size'] = args.batch_size
    if args.test_data:
        config['test']['test_data'] = args.test_data
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    # if not os.path.exists(args.checkpoint):
    #     raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Determine which tester to use 
    if args.tester == "esm_msa_transformer_baseline_ablation":
        tester = ESM_MSATransformer_AblationTester(config, args.checkpoint)
    elif args.tester == "evosite_combined":
        tester = EvoSiteCombinedTester(config, args.checkpoint)
    elif args.tester == "evosite_beta":
        tester = EvoSiteBetaTester(config, args.checkpoint)
    elif args.tester == "evosite_gamma":
        tester = EvoSiteGammaTester(config, args.checkpoint)
    elif args.tester == "evosite_phignet":
        tester = EvoSitePhiGnetTester(config, args.checkpoint)
    elif args.tester == "evosite_betaphignet":
        tester = EvoSiteBetaPhiGnetTester(config, args.checkpoint)
    elif args.tester == "evosite_delta":
        tester = EvoSiteDeltaTester(config, args.checkpoint)
    elif args.tester == "clustersite":
        tester = ClusterSiteTester(config, args)
    else:
        tester = EvoSiteTester(config, args.checkpoint)
    
    try:
        if args.tester == "clustersite":
            tester.test()
        else:
            tester.run_inference()
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
