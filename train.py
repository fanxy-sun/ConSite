#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import logging
import time
import re
import warnings
import math
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
from pathlib import Path
from packaging import version
import pandas as pd
import gc
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
import dgl
# warnings.filterwarnings('ignore')
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Import project modules
from model.EvoSite import (EvoSite, EvoSiteCombined, ESMResidueClassifier, MSATransformerClassifier,
                           EvoSiteBeta, EvoSiteGamma, EvoSitePhiGnet, EvoSiteBetaPhiGnet, EvoSiteGraphEC, EvoSiteDelta,
                           RFDSite, MSASite, RFSite, 
                           ResidueEncoder, ResidueEncoderBeta, ECEmbeddingNet, ActiveSiteEmbeddingNet, ClusterSite,
                           ResidueEncoderGamma, ResidueEncoderAlpha,
                           ClusterSiteBeta, ClusterSiteAlpha, ResidueEncoderGammaBinary, ClusterSiteBinary
                        )
from data_loaders.enzyme_msa_dataloaders import (EnzymeMSADataset, collate_enzyme_msa_batch, create_enzyme_msa_dataloader,
                                                 EnzymeMSAStructureDataset, collate_enzyme_batch,
                                                 MultiPosNeg_dataset_with_mine_EC, MultiPosNeg_dataset_with_mine_SITE, ResidueFeatureDataset, collate_residue_features, EnzymeActiveSiteDataset3,
                                                 GroupedBatchSampler,
                                                 EnzymeActiveSiteBinaryDataset3)
from common.utils import (
    set_seed, setup_logging, log_metrics, compute_binary_classification_metrics, 
    create_loss_function, get_step_based_scheduler, gradient_clipping_with_monitoring, 
    safe_loss_combination, compute_multiple_classification_metrics, pack_residue_feats,
    ConservationWeightedMlmLoss, CoevolutionaryReconstructionLoss, AdaptiveContrastiveLoss,
    SupConHardLoss, 
    read_model_state, move_batch_to_device, build_data_pools, encode_residue_feature,
    evaluate_site_type_prediction, evaluate_ec_prediction, evaluate_site_type_prediction2,
    extract_clean_state_dict, load_pretrain_model_state, evaluate_site_type_prediction_binary
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


class EvoSiteTrainer:

    def __init__(self, config: Dict[str, Any], local_rank: int = -1):
        """
        Initialize trainer with configuration and multi-GPU support.
        
        Args:
            config: Training configuration dictionary
            local_rank: Local rank for distributed training (-1 for single GPU)
        """
        self.config = config
        self.model_config = config['model']
        self.train_config = config['train']
        self.local_rank = local_rank
        

        # ============================分布式训练环境设置 (Distributed Training Environment Setup)============================
        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.is_distributed = True
        else:
            self.local_rank = local_rank
            self.is_distributed = local_rank >= 0
            
        if self.is_distributed:
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.rank = int(os.environ.get('RANK', 0))
        else:
            self.world_size = 1
            self.rank = 0
    
        self.is_main_process = (not self.is_distributed) or (self.rank == 0)


        # ============================日志和目录设置 (Logging and Directory Setup)===========================================
        if self.is_main_process:
            # Setup logging
            os.makedirs(self.train_config['log_dir'], exist_ok=True)
            log_file = os.path.join(self.train_config['log_dir'], 'train.log')
            self.logger = setup_logging(log_file)
            
            # Create output directories
            os.makedirs(self.train_config['output_dir'], exist_ok=True)
            os.makedirs(self.train_config['checkpoint_dir'], exist_ok=True)
        else:
            # Create a dummy logger for non-main processes
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
            

        # ============================设备设置和进程组初始化 (Device Setup and Process Group Initialization)============================    
        if self.is_distributed:
            dist.init_process_group(backend='nccl')
            self._device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
            if self.is_main_process:
                self.logger.info(f"Distributed training enabled. Using device: cuda:{self.local_rank}")
        else:
            self._device = torch.device(self.train_config['device'])
            if self._device.type == 'cuda':
                if self._device.index is not None:
                    torch.cuda.set_device(self._device.index)
                else:
                    torch.cuda.set_device(0)  # 默认使用第一个GPU
                self.logger.info(f"Using device: {self._device} (CUDA device index: {self._device.index})")
        
        # Set seed for reproducibility (important for distributed training)
        set_seed(self.train_config['seed'] + self.rank)
        
        # Initialize mixed precision scaler
        self.use_amp = self.train_config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0      # 当前训练的 epoch 数，从 0 开始计数
        self.best_val_metric = 0.0  # 当前训练过程中验证集（validation set）上取得的最佳指标值（比如 F1 分数或准确率）
        self.patience_counter = 0   # 表示早停（early stopping）计数器，初始为 0。每当验证集指标没有提升时，这个计数器加 1；如果连续多次没有提升，达到设定的 patience（容忍次数），则提前终止训练，防止过拟合和资源浪费。
        self.global_step = 0        # global_step 为何按 accumulation_steps 计，代表一次真实的参数更新（optimizer.step），而非物理 batch 数。
        

        # ============================核心组件初始化 (Core Component Initialization)================================================
        self.stability_config = self.config.get('stability', {})
        self._setup_stability_monitoring()
        
        # Initialize model, data, optimizer
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_loss_functions()
        
        self._setup_optimizations()
        self._update_dca_scaling_scheduler()



    def _update_dca_scaling_scheduler(self):
        """Update DCA scaling scheduler with dynamically calculated total steps."""
        # Check if DCA scaling is enabled
        if not self.model_config.get('use_dynamic_dca_scaling', False):
            return
            
        # Calculate actual total training steps
        train_num_samples = self.train_config.get('train_num_samples', None)
        if train_num_samples is None:
            # Use actual dataset size if not specified
            train_num_samples = len(self.train_dataset)
            if self.is_main_process:
                self.logger.info(f"Using actual dataset size for DCA scaling: {train_num_samples} samples")
        else:
            if self.is_main_process:
                self.logger.info(f"Using configured train_num_samples for DCA scaling: {train_num_samples} samples")
                
        batch_size = self.train_config['batch_size']
        accumulation_steps = self.train_config.get('accumulation_steps', 1)
        epochs = self.train_config['epochs']
        world_size = self.world_size if self.is_distributed else 1
        
        # Calculate steps considering distributed training and gradient accumulation
        batches_per_gpu_per_epoch = math.ceil(train_num_samples / (batch_size * world_size))
        optimizer_steps_per_epoch = math.ceil(batches_per_gpu_per_epoch / accumulation_steps)
        total_optimizer_steps = optimizer_steps_per_epoch * epochs
        
        # Get scaling configuration
        scaling_coverage_ratio = self.train_config.get('scaling_coverage_ratio', 0.8)
        min_scaling_steps = self.train_config.get('min_scaling_steps', 1000)
        max_scaling_steps = self.train_config.get('max_scaling_steps', None)
        
        # Calculate dynamic scaling steps
        dynamic_scaling_steps = int(total_optimizer_steps * scaling_coverage_ratio)
        dynamic_scaling_steps = max(dynamic_scaling_steps, min_scaling_steps)
        if max_scaling_steps is not None:
            dynamic_scaling_steps = min(dynamic_scaling_steps, max_scaling_steps)
        
        # Update the model's DCA scheduler if it exists
        model_to_check = self.model.module if self.is_distributed else self.model
        if hasattr(model_to_check, 'dca_scaling_scheduler') and model_to_check.dca_scaling_scheduler is not None:
            # Update the scheduler's total_steps
            model_to_check.dca_scaling_scheduler.update_total_steps(dynamic_scaling_steps)
            model_to_check.dca_scaling_scheduler.reset()  # Reset current step
            
            if self.is_main_process:
                self.logger.info(f"DCA scaling scheduler updated:")
                self.logger.info(f"  Calculated dynamic steps: {dynamic_scaling_steps}")
                self.logger.info(f"  Coverage ratio: {scaling_coverage_ratio:.1%}")
                self.logger.info(f"  Total training steps: {total_optimizer_steps}")
                self.logger.info(f"  Scheduler state: {model_to_check.dca_scaling_scheduler}")
        else:
            if self.is_main_process:
                self.logger.info(f"No DCA scaling scheduler found in model, skipping dynamic update")



    def _setup_stability_monitoring(self):
        """Initialize numerical stability monitoring."""
        self.nan_count = 0
        self.consecutive_nan_count = 0
        self.total_nan_batches = 0
        self.last_valid_loss = None
        
        self.debug_mode = self.stability_config.get('debug_mode', False)
        self.nan_patience = self.stability_config.get('nan_patience', 5)
        self.nan_recovery = self.stability_config.get('nan_recovery', 'skip')
        self.gradient_clip = self.stability_config.get('gradient_clip', 1.0)
        self.monitor_gradients = self.stability_config.get('monitor_gradients', True)

        self.logger.info(f"Stability monitoring initialized:")
        self.logger.info(f"  - Debug mode: {self.debug_mode}")
        self.logger.info(f"  - NaN patience: {self.nan_patience}")
        self.logger.info(f"  - NaN recovery strategy: {self.nan_recovery}")
        self.logger.info(f"  - Gradient clip threshold: {self.gradient_clip}")
        self.logger.info(f"  - Monitor gradients: {self.monitor_gradients}")
        


    def _setup_optimizations(self):
        if self.is_main_process:
            self.logger.info("Setting up training environment optimizations...")
        
        if torch.cuda.is_available():
            try:
                # cuDNN settings
                enable_cudnn_benchmark = self.train_config.get('cudnn_benchmark', True)
                if enable_cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True
                
                # Attention backend settings (for PyTorch 2.x+)
                use_efficient_backends = self.train_config.get('use_efficient_attention_backends', True)
                if use_efficient_backends:
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                else:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                
                # TF32 precision settings (for Ampere GPUs+)
                use_tf32 = self.train_config.get('use_tf32', True)
                if use_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"部分训练环境优化设置失败: {e}")
        
        # Memory management settings
        self.enable_memory_optimization = self.train_config.get('enable_memory_optimization', True)
        self.memory_cleanup_frequency = self.train_config.get('memory_cleanup_frequency', 50)

        

    def _perform_memory_cleanup(self, batch_idx: int):
        if not self.enable_memory_optimization:
            return
            
        # 定期清理GPU缓存
        if batch_idx % self.memory_cleanup_frequency == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 记录内存使用情况（仅主进程）
                if self.is_main_process and batch_idx % (self.memory_cleanup_frequency * 4) == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                    self.logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")



    def _setup_model(self):
        """Initialize the EvoSite model with multi-GPU support."""
        if self.is_main_process:
            self.logger.info("Initializing EvoSite model...")
        
        # Add device to model config
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        # Initialize model
        self.model = EvoSite(**model_config)
        self.model = self.model.to(self._device)

        if hasattr(torch, 'compile') and self.train_config.get('use_torch_compile', False):
            try:
                compile_mode = self.train_config.get('torch_compile_mode', 'reduce-overhead')
                self.model = torch.compile(self.model, mode=compile_mode)
                if self.is_main_process:
                    self.logger.info(f"torch.compile启用: {compile_mode}模式")
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"torch.compile失败: {e}")

        # Wrap model with DDP for distributed training
        if self.is_distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True  # Needed for complex models like EvoSite
            )
        
        # Log model parameters (only on main process)
        if self.is_main_process:
            model_for_params = self.model.module if self.is_distributed else self.model
            total_params = sum(p.numel() for p in model_for_params.parameters())
            trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
            
            self.logger.info(f"Model initialized with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            if self.is_distributed:
                self.logger.info(f"Using DistributedDataParallel with {self.world_size} GPUs")
            
            config_path = os.path.join(self.train_config['output_dir'], 'model_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.model_config, f, default_flow_style=False)
    


    def _setup_data(self):
        """Setup training and validation datasets."""
        self.logger.info("Setting up datasets...")
        
        # Training dataset
        self.train_dataset = EnzymeMSADataset(
            csv_file=self.train_config['train_data'],
            msa_dir=self.train_config['msa_dir'],
            dca_dir=self.train_config['dca_dir'],
            max_seq_length=self.train_config['max_seq_length'],
            max_msa_sequences=self.train_config['max_msa_sequences'],
            split="train",
            esm_model_name=self.model_config['esm_model_name'],
            msa_model_name=self.model_config['msa_model_name'],
            train_num_samples=self.train_config.get('train_num_samples', None)
        )
        
        # Validation dataset
        self.val_dataset = EnzymeMSADataset(
            csv_file=self.train_config['val_data'],
            msa_dir=self.train_config['msa_dir'],
            dca_dir=self.train_config['dca_dir'],
            max_seq_length=self.train_config['max_seq_length'],
            max_msa_sequences=self.train_config['max_msa_sequences'],
            split="val",
            esm_model_name=self.model_config['esm_model_name'],
            msa_model_name=self.model_config['msa_model_name']
        )
        
        # Setup distributed samplers if using multi-GPU training
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False
            )
        else:
            self.train_sampler = None
            self.val_sampler = None
        
        # Data loaders with distributed sampler support
        if self.is_distributed:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_config['batch_size'],
                sampler=self.train_sampler,
                num_workers=self.train_config['num_workers'],
                pin_memory=self.train_config['pin_memory'],
                persistent_workers=self.train_config.get('persistent_workers', True),
                drop_last=True,
                collate_fn=collate_enzyme_msa_batch,
                prefetch_factor=2 if self.train_config['num_workers'] > 0 else None
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.train_config['batch_size'],
                sampler=self.val_sampler,
                num_workers=self.train_config['num_workers'],
                pin_memory=self.train_config['pin_memory'],
                persistent_workers=self.train_config.get('persistent_workers', True),
                drop_last=False,
                collate_fn=collate_enzyme_msa_batch,
                prefetch_factor=2 if self.train_config['num_workers'] > 0 else None
            )
        else:
            # For single GPU training, use the original helper function
            self.train_loader = create_enzyme_msa_dataloader(
                self.train_dataset,
                batch_size=self.train_config['batch_size'],
                shuffle=True,
                num_workers=self.train_config['num_workers'],
                pin_memory=self.train_config['pin_memory'],
                persistent_workers=self.train_config.get('persistent_workers', True),
                drop_last=True
            )
            
            self.val_loader = create_enzyme_msa_dataloader(
                self.val_dataset,
                batch_size=self.train_config['batch_size'],
                shuffle=False,
                num_workers=self.train_config['num_workers'],
                pin_memory=self.train_config['pin_memory'],
                persistent_workers=self.train_config.get('persistent_workers', True),
                drop_last=False
            )
        
        if self.is_main_process:
            self.logger.info(f"Training samples: {len(self.train_dataset)}")
            self.logger.info(f"Validation samples: {len(self.val_dataset)}")
            self.logger.info(f"Training batches per GPU: {len(self.train_loader)}")
            self.logger.info(f"Validation batches per GPU: {len(self.val_loader)}")
            if self.is_distributed:
                self.logger.info(f"Total training batches across {self.world_size} GPUs: {len(self.train_loader) * self.world_size}")
                self.logger.info(f"Total validation batches across {self.world_size} GPUs: {len(self.val_loader) * self.world_size}")
    

    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler with detailed parameter logging."""
        
        # =========================================== Top@PARAMETER ANALYSIS ======================================================
        self.logger.info("Analyzing model parameters...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = total_params - trainable_count
        
        self.logger.info(f"OVERALL PARAMETER STATISTICS:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_count:,}")
        self.logger.info(f"  Frozen parameters: {frozen_count:,}")
        self.logger.info(f"  Trainable ratio: {trainable_count/total_params:.2%}")
        
        trainable_params_by_component = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                component = name.split('.')[0]
                if component not in trainable_params_by_component:
                    trainable_params_by_component[component] = []
                trainable_params_by_component[component].append(name)
        for component, param_names in sorted(trainable_params_by_component.items()):
            self.logger.info(f"  {component.upper()} - {len(param_names)} parameters:")
            for name in param_names:
                self.logger.info(f"    - {name}")
        # =========================================== Bottom@PARAMETER ANALYSIS ===================================================
        

        # Setup optimizer
        optimizer_type = self.train_config.get('optimizer', 'adamw').lower()
        lr = float(self.train_config['learning_rate'])
        weight_decay = float(self.train_config['weight_decay'])
        
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        

        # Setup scheduler (step-based for better control with gradient accumulation)
        scheduler_type = self.train_config.get('scheduler', 'cosine')
        if scheduler_type != 'none':
            # Calculate total training steps considering gradient accumulation
            steps_per_epoch = len(self.train_loader) // self.train_config.get('accumulation_steps', 1)
            total_training_steps = steps_per_epoch * self.train_config['epochs']
            warmup_steps = int(steps_per_epoch * self.train_config.get('warmup_epochs', 0))
            self.scheduler = get_step_based_scheduler(
                self.optimizer,
                scheduler_type=scheduler_type,
                total_steps=total_training_steps,
                warmup_steps=warmup_steps,
                min_lr=self.train_config.get('min_lr', 1e-6)
            )
            
            self.logger.info(f"  Steps per epoch: {steps_per_epoch}")
            self.logger.info(f"  Total training steps: {total_training_steps}")
            self.logger.info(f"  Warmup steps: {warmup_steps}")
        else:
            self.scheduler = None
        
        # log
        self.logger.info(f"\nOPTIMIZER SETUP:")
        self.logger.info(f"  Optimizer: {optimizer_type}")
        self.logger.info(f"  Learning rate: {lr}")
        self.logger.info(f"  Weight decay: {weight_decay}")
        self.logger.info(f"  Scheduler: {scheduler_type}")
        self.logger.info(f"  Parameters in optimizer: {len(trainable_params)} tensors")
        self.logger.info(f"  Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    


    def _setup_loss_functions(self):
        """Setup loss functions for training."""
        # Main prediction loss
        self.criterion = create_loss_function(
            loss_type=self.train_config.get('loss_type', 'ce'),
            class_weights=self.train_config.get('class_weights', None),
            label_smoothing=self.train_config.get('label_smoothing', 0.0)
        )
        
        # Loss weights
        self.main_loss_weight = self.train_config.get('main_loss_weight', 1.0)
        
        # Only setup weights for enabled features
        if self.model_config.get('use_auxiliary_losses', False):
            self.auxiliary_loss_weight = self.train_config.get('auxiliary_loss_weight', 0.1)
        else:
            self.auxiliary_loss_weight = 0.0  
        
        # ESM-MSA alignment loss (independent of auxiliary losses)
        if self.model_config.get('use_msa_featurizer', False):
            self.alignment_loss_weight = self.train_config.get('alignment_loss_weight', 0.1)
        else:
            self.alignment_loss_weight = 0.0  
            
        # Single site potential loss weight
        if self.model_config.get('use_single_site_potentials', False):
            self.single_site_weight = self.train_config.get('single_site_weight', 0.5)
        else:
            self.single_site_weight = 0.0  
            
        # Range penalty weight (always applied during training if model supports it)
        self.range_penalty_weight = self.train_config.get('range_penalty_weight', 3e-3)
        
        # Prompt-based prediction loss weights
        if self.model_config.get('use_prompt_predictor', False):
            self.prompt_regularization_weight = self.train_config.get('prompt_regularization_weight', 1.0)
            self.prompt_diversity_weight = self.train_config.get('prompt_diversity_weight', 1.0)
            
            # Auxiliary prompt loss for auxiliary mode
            if self.model_config.get('prompt_predictor_mode', 'replace') == 'auxiliary':
                self.prompt_auxiliary_loss_weight = self.train_config.get('prompt_auxiliary_loss_weight', 0.5)
            else:
                self.prompt_auxiliary_loss_weight = 0.0
        else:
            self.prompt_regularization_weight = 0.0
            self.prompt_diversity_weight = 0.0
            self.prompt_auxiliary_loss_weight = 0.0
        
        # Move loss function weights to device if they exist
        if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(self._device)
        if hasattr(self.criterion, 'pos_weight') and self.criterion.pos_weight is not None:
            self.criterion.pos_weight = self.criterion.pos_weight.to(self._device)
        
        self.logger.info(f"Loss functions initialized:")
        self.logger.info(f"  Main loss weight: {self.main_loss_weight}")
        self.logger.info(f"  Auxiliary loss weight: {self.auxiliary_loss_weight}")
        self.logger.info(f"  Alignment loss weight: {self.alignment_loss_weight}")
        self.logger.info(f"  Single site weight: {self.single_site_weight}")
        self.logger.info(f"  Range penalty weight: {self.range_penalty_weight}")
        if self.model_config.get('use_prompt_predictor', False):
            self.logger.info(f"  Prompt regularization weight: {self.prompt_regularization_weight}")
            self.logger.info(f"  Prompt diversity weight: {self.prompt_diversity_weight}")
            if self.prompt_auxiliary_loss_weight > 0:
                self.logger.info(f"  Prompt auxiliary loss weight: {self.prompt_auxiliary_loss_weight}")
    


    def compute_loss(
        self, 
        outputs: Dict[str, Union[torch.Tensor, List[torch.Tensor], Dict[int, torch.Tensor], Any]], 
        batch: Dict[str, Union[torch.Tensor, List[torch.Tensor], List[str], List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss from model outputs and batch labels.
        
        Args:
            outputs: Model outputs dictionary
            batch: Batch data dictionary
            
        Returns:
            Dictionary containing individual and total losses
                losses['main_loss']: token level loss, Average loss across all valid tokens in batch, 由于PyTorch中nn.CrossEntropyLoss的默认reduction参数为'mean'，因此main_loss已经是batch内所有样本的所有token的平均损失
                losses['auxiliary_loss']: sample level loss，Average loss across batch
                losses['alignment_loss']: sample level loss，Average loss across batch
                losses['prompt_regularization_loss']: sample level loss，Prompt regularization loss (if enabled)
                losses['prompt_diversity_loss']: sample level loss，Prompt diversity loss (if enabled)
                losses['prompt_auxiliary_loss']: sample level loss，Auxiliary prompt prediction loss (if in auxiliary mode)
                losses['total_loss']: sample level loss，将前面的losses['main_loss'], losses['auxiliary_loss'], losses['alignment_loss']加权求和，得到losses['total_loss']
        """
        losses = {}
        
        # Main prediction loss
        logits = outputs['logits']  # [batch_size, seq_len, num_classes]
        labels = batch['labels']    # [batch_size, seq_len]
        
        # Apply valid token mask to compute loss only on valid residues
        if 'esm_valid_tokens_mask' in outputs:
            valid_mask = outputs['esm_valid_tokens_mask']  # [batch_size, seq_len]
            
            # Flatten and apply mask
            valid_logits = logits[valid_mask]  # [num_valid_tokens, num_classes]
            valid_labels = labels[valid_mask]  # [num_valid_tokens]
            
            # Let autocast choose precision automatically
            main_loss = self.criterion(valid_logits, valid_labels)
        else:
            raise RuntimeError('esm_valid_tokens_mask not in outputs')
        
        losses['main_loss'] = main_loss
        
        # Additional loss components
        additional_loss_components = []
        
        # Auxiliary losses ONLY refer to ESM-DCA coupling matrix alignment losses
        if self.model_config.get('use_auxiliary_losses', False) and 'auxiliary_loss' in outputs:
            aux_loss = outputs['auxiliary_loss']
            if self._check_loss_validity(aux_loss, "auxiliary_loss"):
                losses['auxiliary_loss'] = aux_loss
                additional_loss_components.append(losses['auxiliary_loss'] * self.auxiliary_loss_weight)
        
        # MSA-ESM alignment loss
        if (self.model_config.get('fusion_method', '') == "soft_label" and 
            self.model_config.get('use_msa_featurizer', False) and 
            'residue_level_soft_label_loss' in outputs):
            align_loss = outputs['residue_level_soft_label_loss']
            if self._check_loss_validity(align_loss, "alignment_loss"):
                losses['alignment_loss'] = align_loss
                additional_loss_components.append(losses['alignment_loss'] * self.alignment_loss_weight)
        
        # Single-site potential loss 
        if self.model_config.get('use_single_site_potentials', False) and 'single_site_loss' in outputs:
            ss_loss = outputs['single_site_loss']
            if self._check_loss_validity(ss_loss, "single_site_loss"):
                losses['single_site_loss'] = ss_loss
                additional_loss_components.append(losses['single_site_loss'] * self.single_site_weight)
        
        # Range penalty loss
        if self.model.training and 'range_penalty_loss' in outputs:
            rp_loss = outputs['range_penalty_loss']
            if self._check_loss_validity(rp_loss, "range_penalty_loss"):
                losses['range_penalty_loss'] = rp_loss
                additional_loss_components.append(losses['range_penalty_loss'] * self.range_penalty_weight)
        
        # Prompt-based prediction losses
        if self.model_config.get('use_prompt_predictor', False):
            # Prompt regularization loss
            if 'prompt_regularization_loss' in outputs:
                prompt_reg_loss = outputs['prompt_regularization_loss']
                if self._check_loss_validity(prompt_reg_loss, "prompt_regularization_loss"):
                    losses['prompt_regularization_loss'] = prompt_reg_loss
                    additional_loss_components.append(losses['prompt_regularization_loss'] * self.prompt_regularization_weight)
            
            # Prompt diversity loss
            if 'prompt_diversity_loss' in outputs:
                prompt_div_loss = outputs['prompt_diversity_loss']
                if self._check_loss_validity(prompt_div_loss, "prompt_diversity_loss"):
                    losses['prompt_diversity_loss'] = prompt_div_loss
                    additional_loss_components.append(losses['prompt_diversity_loss'] * self.prompt_diversity_weight)
            
            # Auxiliary prompt loss (when prompt is used as auxiliary prediction)
            if (self.model_config.get('prompt_predictor_mode', 'replace') == 'auxiliary' and 
                'prompt_logits' in outputs and self.prompt_auxiliary_loss_weight > 0):
                
                prompt_logits = outputs['prompt_logits']  # [batch_size, seq_len, num_classes]
                
                # Apply same masking as main loss
                valid_prompt_logits = prompt_logits[valid_mask]  # [num_valid_tokens, num_classes]
                valid_labels_for_prompt = labels[valid_mask]  # [num_valid_tokens]
                
                # Let autocast choose precision automatically
                prompt_aux_loss = self.criterion(valid_prompt_logits, valid_labels_for_prompt)
                
                if self._check_loss_validity(prompt_aux_loss, "prompt_auxiliary_loss"):
                    losses['prompt_auxiliary_loss'] = prompt_aux_loss
                    additional_loss_components.append(losses['prompt_auxiliary_loss'] * self.prompt_auxiliary_loss_weight)
        
        # loss combination - let autocast handle precision
        if self.stability_config.get('safe_loss_combination', True):
            main_loss_weighted = losses['main_loss'] * self.main_loss_weight
            all_losses = [main_loss_weighted] + additional_loss_components
            total_loss = safe_loss_combination(*all_losses)
        else:
            # Traditional combination - let autocast handle precision
            total_loss = losses['main_loss'] * self.main_loss_weight
            for additional_loss in additional_loss_components:
                total_loss = total_loss + additional_loss
        
        # Final loss validation
        if not self._check_loss_validity(total_loss, "total_loss"):
            self.logger.warning(f"Invalid total loss encountered: {total_loss}")
            if self.nan_recovery == "skip":
                # Return a dummy loss that won't update gradients
                total_loss = torch.zeros(1, device=self._device, requires_grad=True)
            else:
                raise RuntimeError("Invalid total loss encountered")
        
        losses['total_loss'] = total_loss
        
        return losses
    


    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        
        total_losses = {}
        num_batches = len(self.train_loader)
        accumulation_steps = self.train_config.get('accumulation_steps', 1)
        log_frequency = self.train_config.get('log_frequency', 10)
        
        # Create progress bar only for main process
        if self.is_main_process:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.current_epoch + 1}/{self.train_config['epochs']} [TRAIN]",
                unit="batch"
            )
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            batch_failed = False    # 分布式训练异常处理标志
            try:
                batch = self._move_batch_to_device(batch)
                
                # Forward with mixed precision - let autocast choose precision
                with autocast(enabled=self.use_amp):
                    outputs = self.model(batch)
                    losses = self.compute_loss(outputs, batch)      # losses是字典，每个键对应一种损失，每个值是一个标量是sample level loss，以'sample level loss' losses['total_loss']进行反向传播
                    
                    # Scale loss by accumulation steps
                    loss = losses['total_loss'].float() / accumulation_steps
                    
            except Exception as e:
                batch_failed = True
                
                # Log detailed error information
                sample_ids = batch.get('sample_id', ['unknown'] * len(batch.get('protein_ids', ['unknown'])))
                protein_ids = batch.get('protein_ids', ['unknown'] * len(sample_ids))
                self.logger.warning(f"训练批次 {batch_idx} 前向传播失败 - 跳过该批次")
                self.logger.warning(f"错误类型: {type(e).__name__}")
                self.logger.warning(f"错误信息: {str(e)}")
                self.logger.warning(f"问题样本: {list(zip(sample_ids, protein_ids))}")
                self.logger.warning(f"批次大小: {len(sample_ids)}")
                
                # 强制刷新输出缓冲区
                sys.stdout.flush()
                sys.stderr.flush()
            
            # 在分布式训练中，同步异常处理状态（仅在有异常时才进行通信）
            if self.is_distributed:
                batch_failed_tensor = torch.tensor(int(batch_failed), device=self._device)
                dist.all_reduce(batch_failed_tensor, op=dist.ReduceOp.MAX)
                if batch_failed_tensor.item() > 0:      # 如果任何一个进程失败，所有进程都跳过这个批次
                    if not batch_failed and self.is_main_process:
                        self.logger.warning(f"批次 {batch_idx}: 其他进程失败，主进程同步跳过")
                    continue
            elif batch_failed:
                continue        # 单GPU训练直接跳过
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate losses
            for name, loss_val in losses.items():
                if name not in total_losses:
                    total_losses[name] = 0.0
                total_losses[name] += loss_val.item()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if self.use_amp:
                    # Enhanced gradient clipping with monitoring
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        if self.monitor_gradients:
                            grad_norm = gradient_clipping_with_monitoring(self.model, self.gradient_clip)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Enhanced gradient clipping with monitoring
                    if self.gradient_clip > 0:
                        if self.monitor_gradients:
                            grad_norm = gradient_clipping_with_monitoring(self.model, self.gradient_clip)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.optimizer.step()
                
                # Debug output for monitoring
                if self.debug_mode and batch_idx % 100 == 0:
                    nan_stats = self._get_nan_stats()
                    self.logger.info(f"Batch {batch_idx}: NaN stats = {nan_stats}, grad_norm = {grad_norm:.4f}")
                
                # Step-based learning rate scheduler and DCA scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                # Update DCA scaling factor AFTER the optimizer step
                # Correctly access the model instance, whether wrapped by DDP or not
                model_instance = self.model.module if self.is_distributed else self.model
                if model_instance.use_dynamic_dca_scaling and model_instance.dca_scaling_scheduler is not None:
                    model_instance.dca_scaling_scheduler.step()  # 直接调用调度器的step方法
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Perform memory cleanup
                self._perform_memory_cleanup(batch_idx)
            
            # Update progress bar (only on main process)
            if batch_idx % log_frequency == 0 and self.is_main_process:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",           # 每个batch处理后，进度条显示的是sample level loss，详见compute_loss函数的Returns
                    'lr': f"{current_lr:.2e}"
                })
        
        # 计算平均损失, epoch_losses是字典，每个键对应一种损失，每个值是一个浮点数，该浮点数是sample level loss(先在一个batch内求平均值，然后对num_batches个'batch平均值'累加，然后再除以num_batches求平均)
        epoch_losses = {name: loss / num_batches for name, loss in total_losses.items()}
        
        return epoch_losses
    


    @torch.no_grad()
    def validate_epoch(
        self,
        evaluate_num_samples: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:

        self.model.eval()
        total_losses = {}
        all_metrics = defaultdict(list)
        
        val_loader = self.val_loader
        if evaluate_num_samples is not None:
            batch_size = self.train_config['batch_size']
            target_batches = (evaluate_num_samples + batch_size - 1) // batch_size  # 向上取整
            target_batches = min(target_batches, len(val_loader))
        else:
            target_batches = len(val_loader)
        
        # Create progress bar only for main process
        if self.is_main_process:
            pbar = tqdm(
                enumerate(val_loader),
                total=target_batches,
                desc=f"Epoch {self.current_epoch + 1}/{self.train_config['epochs']} [VAL]",
                unit="batch"
            )
        else:
            pbar = enumerate(val_loader)
        
        try:
            for batch_idx, batch in pbar:
                if batch_idx >= target_batches:
                    break
                batch_failed = False
                
                try:
                    batch = self._move_batch_to_device(batch)
                    with autocast(enabled=False):   # 验证阶段明确禁用autocast，确保使用float32避免类型不匹配
                        outputs = self.model(batch)
                        losses = self.compute_loss(outputs, batch)      # sample level loss
                except Exception as e:
                    batch_failed = True
                    sample_ids = batch.get('sample_id', ['unknown'] * len(batch.get('protein_ids', ['unknown'])))
                    protein_ids = batch.get('protein_ids', ['unknown'] * len(sample_ids))
                    self.logger.warning(f"验证批次 {batch_idx} 前向传播失败 - 跳过该批次")
                    self.logger.warning(f"错误类型: {type(e).__name__}")
                    self.logger.warning(f"错误信息: {str(e)}")
                    self.logger.warning(f"问题样本: {list(zip(sample_ids, protein_ids))}")
                    self.logger.warning(f"批次大小: {len(sample_ids)}")
                
                if self.is_distributed:
                    batch_failed_tensor = torch.tensor(int(batch_failed), device=self._device)
                    dist.all_reduce(batch_failed_tensor, op=dist.ReduceOp.MAX)
                    
                    # 多GPU训练，如果任何一个进程失败，所有进程都跳过这个批次
                    if batch_failed_tensor.item() > 0:
                        if not batch_failed and self.is_main_process:
                            self.logger.warning(f"验证批次 {batch_idx}: 其他进程失败，主进程同步跳过")
                        continue
                elif batch_failed:
                    # 单GPU训练直接跳过
                    continue
                

                for name, loss_val in losses.items():
                    if name not in total_losses:
                        total_losses[name] = 0.0
                    total_losses[name] += loss_val.item()       # total_losses字典的每个键对应一种损失，每个值是一个浮点数，该浮点数对sample level loss（即已经在一个batch内取平均值）累加
                

                logits = outputs['logits']  # [batch_size, seq_len, num_classes]
                labels = batch['labels']    # [batch_size, seq_len]
                
                if 'esm_valid_tokens_mask' in outputs:
                    valid_mask = outputs['esm_valid_tokens_mask']
                else:
                    raise RuntimeError('esm_valid_tokens_mask not in outputs')
                

                probs = torch.softmax(logits, dim=-1)
                if self.model_config['classification_output_dim'] == 2:
                    preds = torch.argmax(probs, dim=-1)
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
                    # 多分类处理
                    preds = torch.argmax(probs, dim=-1)
                    num_classes = self.model_config['classification_output_dim']
                    
                    # 1 计算二分类指标（活性位点vs非活性位点），计算方式与EasIFA一致
                    binary_labels = (labels != 0).float()
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
                    

                    # 2 计算多分类指标，计算方式与EasIFA不一致
                    for site_type in range(1, num_classes):                 # 跳过背景类(class 0)
                        binary_labels = (labels == site_type).float()       # [batch_size, seq_len]，标记出哪些位置的真实类别等于当前的 site_type
                        binary_preds = (preds == site_type).float()         # [batch_size, seq_len]，标记哪些位置被预测为当前的 site_type。
                        site_prob = probs[:, :, site_type]                  # [batch_size, seq_len]，从模型输出的概率分布 probs 中，抽取当前类别 site_type 的预测概率。
                        
                        # 如果当前批次中不存在此类，则跳过
                        if binary_labels.sum() == 0:
                            continue
                        
                        site_metrics = compute_binary_classification_metrics(
                            predictions=binary_preds,
                            probabilities=site_prob,
                            labels=binary_labels,
                            valid_mask=valid_mask,
                            metrics=['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc', 'ap', 'mcc']
                        )
                        
                        for metric_name, metric_values in site_metrics.items():
                            all_metrics[f"pre_type_{site_type}_{metric_name}"].extend(metric_values)
                    

                    # 3 计算多分类指标，计算方式与EasIFA一致
                    multi_metrics = compute_multiple_classification_metrics(
                        pred=preds,
                        gt=labels,
                        num_site_types=num_classes,
                        valid_mask=valid_mask
                    )
                    
                    for metric_name, values in multi_metrics.items():
                        if values:  # 确保有值
                            # 将metric_name中的连字符替换为下划线，保持命名一致性
                            metric_key = metric_name.replace('-', '_')
                            all_metrics[metric_key].extend(values)
        finally:
            # Only close progress bar if it's a tqdm object (main process)
            if hasattr(pbar, 'close'):
                pbar.close()
        
        # 计算平均损失, epoch_losses是字典，每个键对应一种损失，每个值是一个浮点数，该浮点数是sample level loss(先在一个batch内求平均值，然后对target_batches个'batch平均值'累加，然后再除以target_batches求平均)
        if target_batches > 0 and total_losses:
            epoch_losses = {name: loss / target_batches for name, loss in total_losses.items()}
        else:
            epoch_losses = {}
        
        # Synchronize validation losses across GPUs if using distributed training
        if self.is_distributed and epoch_losses:
            # Convert losses to tensors for synchronization
            loss_tensors = {}
            for name, loss in epoch_losses.items():
                loss_tensor = torch.tensor(loss, device=self._device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensors[name] = loss_tensor.item() / self.world_size
            epoch_losses = loss_tensors
        
        # 计算所有批次的平均指标
        classification_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:  # 确保有值
                classification_metrics[metric_name] = sum(values) / len(values)
        
        # Synchronize classification metrics across GPUs if using distributed training
        if self.is_distributed and classification_metrics:
            metric_tensors = {}
            for name, metric in classification_metrics.items():
                metric_tensor = torch.tensor(metric, device=self._device)
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                metric_tensors[name] = metric_tensor.item() / self.world_size
            classification_metrics = metric_tensors
        
        # 计算多分类宏平均指标
        if self.model_config['classification_output_dim'] > 2:
            # 寻找所有类特定指标
            class_metrics = defaultdict(list)
            for metric_name in list(classification_metrics.keys()):
                if metric_name.startswith('type_') and '_' in metric_name[5:]:
                    parts = metric_name.split('_')
                    if len(parts) >= 3:
                        base_metric = '_'.join(parts[2:])
                        class_metrics[base_metric].append(classification_metrics[metric_name])
            
            # 计算宏平均值
            for metric_name, values in class_metrics.items():
                if values:
                    classification_metrics[f"macro_{metric_name}"] = sum(values) / len(values)
        
        return epoch_losses, classification_metrics



    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model checkpoint
            filename: Optional checkpoint filename (defaults to epoch-based naming)
        """
        if not self.is_main_process:
            return  # Only save checkpoints on the main process
            
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pt"      # Use 1-based epoch numbering for checkpoint filename
        filepath = os.path.join(self.train_config['checkpoint_dir'], filename)

        model_DDP = self.model.module if self.is_distributed else self.model
        model_to_save = getattr(model_DDP, '_orig_mod', model_DDP)

        
        checkpoint = {
            # Model state: core model parameters
            'model_state': {
                'state_dict': model_to_save.state_dict(),
                'model_class': model_to_save.__class__.__name__
            },
            
            # Optimizer state: optimization parameters and momentum
            'optimizer_state': {
                'state_dict': self.optimizer.state_dict(),
                'optimizer_class': self.optimizer.__class__.__name__
            },
            
            # Scheduler state: learning rate scheduling
            'scheduler_state': {
                'state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'scheduler_class': self.scheduler.__class__.__name__ if self.scheduler is not None else None
            },
            
            # Mixed precision state: gradient scaling for AMP
            'scaler_state': {
                'state_dict': self.scaler.state_dict() if self.scaler is not None else None,
                'enabled': self.use_amp
            },
            
            # Training progress state: epoch, metrics, and global counters
            'training_state': {
                'epoch': self.current_epoch,  # Keep internal epoch as 0-based for consistency
                'global_step': self.global_step,
                'best_val_metric': self.best_val_metric,
                'patience_counter': self.patience_counter,
                'current_loss': getattr(self, 'current_val_loss', 0.0)
            },
            
            # Configuration state: model and training configuration
            'config_state': {
                'config': self.config
            },
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        self.logger.info(f"Modular checkpoint saved to {filepath}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.train_config['checkpoint_dir'], "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model checkpoint saved to {best_path}")



    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        

        is_modular = ('model_state' in checkpoint and 'training_state' in checkpoint)
        # New format
        if is_modular:
            model_state_dict = checkpoint['model_state']['state_dict']
            
            if self.optimizer is not None and 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state']['state_dict'])
            
            if self.scheduler is not None and 'scheduler_state' in checkpoint:
                if checkpoint['scheduler_state']['state_dict'] is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state']['state_dict'])
            
            if self.scaler is not None and 'scaler_state' in checkpoint:
                if checkpoint['scaler_state']['state_dict'] is not None:
                    self.scaler.load_state_dict(checkpoint['scaler_state']['state_dict'])
            
            training_state = checkpoint['training_state']
            self.current_epoch = training_state['epoch']
            self.global_step = training_state.get('global_step', 0)
            self.best_val_metric = training_state.get('best_val_metric', 0.0)
            self.patience_counter = training_state.get('patience_counter', 0)
        # Legacy format
        else:
            model_state_dict = checkpoint['model_state_dict']
            
            if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_metric = checkpoint.get('best_metric', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
            self.patience_counter = 0  # Reset for legacy checkpoints
        
        # =========================================== Top@Handle DDP and torch.compile prefix mismatch ======================================================
        current_model_keys = list(self.model.state_dict().keys())
        checkpoint_keys = list(model_state_dict.keys())
        
        if len(current_model_keys) > 0 and len(checkpoint_keys) > 0:
            # Detect wrapper status for the current model
            current_has_ddp = current_model_keys[0].startswith('module.')
            current_has_compile = '_orig_mod.' in current_model_keys[0]

            # Detect wrapper status for the checkpoint
            checkpoint_has_ddp = checkpoint_keys[0].startswith('module.')
            checkpoint_has_compile = '_orig_mod.' in checkpoint_keys[0]

            # Logic to align prefixes. We first create a "clean" state_dict by stripping all known prefixes
            # from the checkpoint, and then add the prefixes required by the current model.
            
            # Step 1: Create a clean state_dict by stripping all prefixes from the checkpoint keys.
            clean_state_dict = {}
            for k, v in model_state_dict.items():
                temp_k = k
                if checkpoint_has_ddp:
                    temp_k = temp_k.replace('module.', '', 1)
                if checkpoint_has_compile:
                    temp_k = temp_k.replace('_orig_mod.', '', 1)
                clean_state_dict[temp_k] = v
            
            # Step 2: Add the necessary prefixes to the clean state_dict to match the current model.
            final_state_dict = {}
            for k, v in clean_state_dict.items():
                temp_k = k
                if current_has_compile:
                    temp_k = '_orig_mod.' + temp_k
                if current_has_ddp:
                    temp_k = 'module.' + temp_k
                final_state_dict[temp_k] = v
            
            model_state_dict = final_state_dict
        # =========================================== Bottom@Handle DDP and torch.compile prefix mismatch ===================================================

        # Load model state, now with correctly aligned keys
        try:
            self.model.load_state_dict(model_state_dict, strict=True)
            self.logger.info(f"Successfully loaded model state_dict from {checkpoint_path}")
        except RuntimeError as e:
            # Enhanced error logging for easier debugging
            self.logger.error(f"Error loading state_dict from {checkpoint_path}: {e}")
            self.logger.error("This may be due to a model architecture mismatch or key-matching failure.")
            current_keys = set(self.model.state_dict().keys())
            loaded_keys = set(model_state_dict.keys())
            missing_keys = current_keys - loaded_keys
            unexpected_keys = loaded_keys - current_keys
            if missing_keys:
                self.logger.error(f"Missing keys in loaded state_dict: {list(missing_keys)[:5]}...")
            if unexpected_keys:
                self.logger.error(f"Unexpected keys in loaded state_dict: {list(unexpected_keys)[:5]}...")
            raise e
        
        # Memory cleanup - release checkpoint data
        del checkpoint, model_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resumed from epoch {self.current_epoch + 1} (global step {self.global_step})")
    


    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.train_config['epochs']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.train_config['epochs']):
            self.current_epoch = epoch
            
            # Set epoch for distributed sampler to ensure different shuffling each epoch
            if self.is_distributed and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            # Train
            train_losses = self.train_epoch()
        
            # Validation 
            val_frequency = self.train_config.get('val_frequency', 1)
            if (epoch + 1) % val_frequency == 0:
                val_losses, val_metrics = self.validate_epoch(evaluate_num_samples=self.train_config.get('evaluate_num_samples'))
                self.current_val_loss = val_losses['total_loss']
                
                # Log metrics
                log_metrics(train_losses, epoch, "train", self.logger)
                log_metrics(val_losses, epoch, "val_loss", self.logger)
                log_metrics(val_metrics, epoch, "val_metrics", self.logger)
                
                # Check for best model
                current_metric = val_metrics.get('multi_class mcc') 
                is_best = current_metric > self.best_val_metric
                
                if is_best:
                    self.best_val_metric = current_metric
                    self.patience_counter = 0
                    self.logger.info(f"New best model! F1: {current_metric:.4f}")
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                save_frequency = self.train_config.get('save_frequency', 1)
                if (epoch + 1) % save_frequency == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
            
            else:
                # Only training, no validation
                log_metrics(train_losses, epoch, "train", self.logger)
            
            # Early stopping
            patience = self.train_config.get('early_stopping_patience', float('inf'))
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping after {patience} epochs without improvement")
                break
        
        # Training completed
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best validation metric: {self.best_val_metric:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(filename="final_checkpoint.pt")



    def _check_loss_validity(self, loss, step_name=""):
        """Check if loss is valid and update NaN statistics."""
        is_valid = not (torch.isnan(loss) or torch.isinf(loss))
        
        if not is_valid:
            self.nan_count += 1
            self.consecutive_nan_count += 1
            self.total_nan_batches += 1
            
            self.logger.warning(f"{step_name}: Invalid loss detected {loss} "
                              f"(consecutive: {self.consecutive_nan_count}, total: {self.total_nan_batches})")
            
            if self.consecutive_nan_count >= self.nan_patience:
                if self.nan_recovery == 'stop':
                    raise RuntimeError(f"Too many consecutive NaN losses ({self.consecutive_nan_count})")
                elif self.nan_recovery == 'reset':
                    self.logger.warning("Resetting consecutive NaN count")
                    self.consecutive_nan_count = 0
        else:
            self.consecutive_nan_count = 0
            self.last_valid_loss = loss.item()
        
        return is_valid



    def _get_nan_stats(self):
        """Get current NaN statistics."""
        return {
            'total_nan_batches': self.total_nan_batches,
            'consecutive_nan_count': self.consecutive_nan_count,
            'last_valid_loss': self.last_valid_loss
        }
    


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

        if isinstance(batch, dgl.DGLGraph):
            return batch.to(self._device)
        
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



class ESM_MSATransformer_AblationTrainer(EvoSiteTrainer):
    """
    Unified trainer for ablation study models (ESM and MSA baselines).
    """
    def __init__(self, config: Dict[str, Any], local_rank: int = -1):
        """
        Initialize ablation trainer with configuration.
        
        Args:
            config: Training configuration dictionary
            local_rank: Local rank for distributed training (-1 for single GPU)
        """
        super().__init__(config, local_rank)



    def _update_dca_scaling_scheduler(self):
        """
        Override to skip DCA scaling scheduler setup for ablation models.
        Ablation models don't use DCA features, so no scaling is needed.
        """
        if self.is_main_process:
            self.logger.info("Skipping DCA scaling scheduler setup for ablation model")
        pass



    def _setup_model(self):
        """Initialize the ablation model based on configuration."""
        if self.is_main_process:
            self.logger.info("Setting up ablation model...")
        
        # Get model type and configuration
        model_type = self.model_config.get('model_type', 'esm')  # 'esm' or 'msa'
        
        try:
            # Create ablation model based on type
            if model_type.lower() == 'esm':
                self.model = ESMResidueClassifier(
                    esm_model_name=self.model_config.get('esm_model_name', "esm2_t33_650M_UR50D"),
                    num_classes=self.model_config.get('classification_output_dim', 4),
                    hidden_dim_multiplier=self.model_config.get('hidden_dim_multiplier', 2),
                    dropout=self.model_config.get('dropout', 0.1),
                    device=self.model_config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
                )
            elif model_type.lower() == 'msa':
                self.model = MSATransformerClassifier(
                    msa_model_name=self.model_config.get('msa_model_name', "esm_msa1b_t12_100M_UR50S"),
                    num_classes=self.model_config.get('classification_output_dim', 4),
                    hidden_dim_multiplier=self.model_config.get('hidden_dim_multiplier', 2),
                    dropout=self.model_config.get('dropout', 0.1),
                    device=self.model_config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
                )
            else:
                raise ValueError(f"Unsupported ablation model type: {model_type}. Supported: 'esm', 'msa'")
            
            if self.is_main_process:
                self.logger.info(f"Created {model_type.upper()} baseline model")
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                self.logger.info(f"Total parameters: {total_params:,}")
                self.logger.info(f"Trainable parameters: {trainable_params:,}")
                self.logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
                self.logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
                
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"Failed to create {model_type} model: {e}")
            raise

        self.model.to(self._device)

        if hasattr(torch, 'compile') and self.train_config.get('use_torch_compile', False):
            try:
                compile_mode = self.train_config.get('torch_compile_mode', 'reduce-overhead')
                self.model = torch.compile(self.model, mode=compile_mode)
                if self.is_main_process:
                    self.logger.info(f"torch.compile enabled with {compile_mode} mode")
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"torch.compile failed: {e}")
        
        # Setup distributed training
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False  # All parameters should be used in ablation models
            )
            
        if self.is_main_process:
            self.logger.info(f"Model moved to device: {self._device}")
            if self.is_distributed:
                self.logger.info(f"Model wrapped with DDP for distributed training")



    def _setup_loss_functions(self):
        """Setup loss functions for training."""
        # Main prediction loss
        self.criterion = create_loss_function(
            loss_type=self.train_config.get('loss_type', 'ce'),
            class_weights=self.train_config.get('class_weights', None),
            label_smoothing=self.train_config.get('label_smoothing', 0.0)
        )
        
        # Loss weights (simplified for ablation models)
        self.main_loss_weight = self.train_config.get('main_loss_weight', 1.0)
        
        # For ablation models, auxiliary losses are disabled
        self.auxiliary_loss_weight = 0.0
        self.alignment_loss_weight = 0.0
        self.single_site_weight = 0.0
        self.range_penalty_weight = 0.0
        
        # Move loss function weights to device if they exist
        if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(self._device)
        if hasattr(self.criterion, 'pos_weight') and self.criterion.pos_weight is not None:
            self.criterion.pos_weight = self.criterion.pos_weight.to(self._device)
        
        self.logger.info(f"Loss functions initialized:")
        self.logger.info(f"  Main loss weight: {self.main_loss_weight}")



    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Union[torch.Tensor, List[torch.Tensor], List[str], List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification loss for ablation models.
        
        Args:
            outputs: Model outputs containing logits
            batch: Batch data containing targets
            
        Returns:
            Dictionary containing computed losses
        """
        losses = {}
        
        # Main prediction loss
        logits = outputs['logits']  # [batch_size, seq_len, num_classes]
        labels = batch['labels']    # [batch_size, seq_len]
        
        # Apply valid token mask to compute loss only on valid residues
        if 'valid_mask' in outputs:
            valid_mask = outputs['valid_mask']  # [batch_size, seq_len]
            
            # Flatten and apply mask
            valid_logits = logits[valid_mask]  # [num_valid_tokens, num_classes]
            valid_labels = labels[valid_mask]  # [num_valid_tokens]
            
            main_loss = self.criterion(valid_logits, valid_labels)
        else:
            raise RuntimeError('valid_mask not in outputs')
        
        losses['main_loss'] = main_loss
        
        # For ablation models, total loss is just the main classification loss
        losses['total_loss'] = main_loss * self.main_loss_weight
        
        return losses



    def _forward_ablation_model(self, batch: Dict[str, Union[torch.Tensor, List[torch.Tensor], List[str], List[int]]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ablation models with proper parameter extraction.
        
        Args:
            batch: Batch dictionary from data loader
            
        Returns:
            Model outputs dictionary
        """
        # Determine model type from config
        model_type = self.model_config.get('model_type', 'esm').lower()
        
        # Generate valid_mask for tokens (ESM-1b style)
        # Based on collate_enzyme_msa_batch token configuration
        padding_idx = 1  # <pad> token index
        cls_idx = 0      # <cls> token index (used as BOS)
        eos_idx = 2      # <eos> token index
        
        if model_type == 'esm':
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
            
        elif model_type == 'msa':
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
            raise ValueError(f"Unsupported ablation model type: {model_type}")
        
        return outputs



    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        total_losses = {}
        num_batches = len(self.train_loader)
        accumulation_steps = self.train_config.get('accumulation_steps', 1)
        log_frequency = self.train_config.get('log_frequency', 10)
        
        # Create progress bar only for main process
        if self.is_main_process:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.current_epoch + 1}/{self.train_config['epochs']} [TRAIN]",
                unit="batch"
            )
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # 分布式训练异常处理标志
            batch_failed = False
            
            try:
                # Properly move all tensors to device, including lists of tensors
                batch = self._move_batch_to_device(batch)
                
                # Forward pass with autocast for proper mixed precision
                with autocast(enabled=self.use_amp):
                    outputs = self._forward_ablation_model(batch)
                    losses = self.compute_loss(outputs, batch)      # losses是字典，每个键对应一种损失，每个值是一个标量是sample level loss，以'sample level loss' losses['total_loss']进行反向传播
                    
                    # Scale loss by accumulation steps
                    loss = losses['total_loss'].float() / accumulation_steps
                    
            except Exception as e:
                batch_failed = True
                
                # Log detailed error information
                sample_ids = batch.get('sample_id', ['unknown'] * len(batch.get('protein_ids', ['unknown'])))
                protein_ids = batch.get('protein_ids', ['unknown'] * len(sample_ids))
                self.logger.warning(f"训练批次 {batch_idx} 前向传播失败 - 跳过该批次")
                self.logger.warning(f"错误类型: {type(e).__name__}")
                self.logger.warning(f"错误信息: {str(e)}")
                self.logger.warning(f"问题样本: {list(zip(sample_ids, protein_ids))}")
                self.logger.warning(f"批次大小: {len(sample_ids)}")
                
                # 强制刷新输出缓冲区
                sys.stdout.flush()
                sys.stderr.flush()
            
            # 在分布式训练中，同步异常处理状态（仅在有异常时才进行通信）
            if self.is_distributed:
                # 使用单次allreduce检查所有进程的异常状态
                batch_failed_tensor = torch.tensor(int(batch_failed), device=self._device)
                dist.all_reduce(batch_failed_tensor, op=dist.ReduceOp.MAX)
                
                # 如果任何一个进程失败，所有进程都跳过这个批次
                if batch_failed_tensor.item() > 0:
                    if not batch_failed and self.is_main_process:
                        self.logger.warning(f"批次 {batch_idx}: 其他进程失败，主进程同步跳过")
                    continue
            elif batch_failed:
                # 单GPU训练直接跳过
                continue
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate losses
            for name, loss_val in losses.items():
                if name not in total_losses:
                    total_losses[name] = 0.0
                total_losses[name] += loss_val.item()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if self.use_amp:
                    # Enhanced gradient clipping with monitoring
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        if self.monitor_gradients:
                            grad_norm = gradient_clipping_with_monitoring(self.model, self.gradient_clip)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Enhanced gradient clipping with monitoring
                    if self.gradient_clip > 0:
                        if self.monitor_gradients:
                            grad_norm = gradient_clipping_with_monitoring(self.model, self.gradient_clip)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.optimizer.step()
                
                # Debug output for monitoring
                if self.debug_mode and batch_idx % 100 == 0:
                    nan_stats = self._get_nan_stats()
                    self.logger.info(f"Batch {batch_idx}: NaN stats = {nan_stats}, grad_norm = {grad_norm:.4f}")
                
                # Step-based learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Perform memory cleanup
                self._perform_memory_cleanup(batch_idx)
            
            # Update progress bar (only on main process)
            if batch_idx % log_frequency == 0 and self.is_main_process:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",           # 每个batch处理后，进度条显示的是sample level loss，详见compute_loss函数的Returns
                    'lr': f"{current_lr:.2e}"
                })
        
        # 计算平均损失, epoch_losses是字典，每个键对应一种损失，每个值是一个浮点数，该浮点数是sample level loss(先在一个batch内求平均值，然后对num_batches个'batch平均值'累加，然后再除以num_batches求平均)
        epoch_losses = {name: loss / num_batches for name, loss in total_losses.items()}
        
        return epoch_losses



    @torch.no_grad()
    def validate_epoch(
        self,
        evaluate_num_samples: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:

        self.model.eval()
        total_losses = {}
        all_metrics = defaultdict(list)
        
        val_loader = self.val_loader
        if evaluate_num_samples is not None:
            batch_size = self.train_config['batch_size']
            target_batches = (evaluate_num_samples + batch_size - 1) // batch_size  # 向上取整
            target_batches = min(target_batches, len(val_loader))
        else:
            target_batches = len(val_loader)
        
        # Create progress bar only for main process
        if self.is_main_process:
            pbar = tqdm(
                enumerate(val_loader),
                total=target_batches,
                desc=f"Epoch {self.current_epoch + 1}/{self.train_config['epochs']} [VAL]",
                unit="batch"
            )
        else:
            pbar = enumerate(val_loader)
        
        try:
            for batch_idx, batch in pbar:
                if batch_idx >= target_batches:
                    break
                batch_failed = False
                
                try:
                    batch = self._move_batch_to_device(batch)
                    with autocast(enabled=self.use_amp):
                        outputs = self._forward_ablation_model(batch)
                        losses = self.compute_loss(outputs, batch)      # sample level loss
                except Exception as e:
                    batch_failed = True
                    sample_ids = batch.get('sample_id', ['unknown'] * len(batch.get('protein_ids', ['unknown'])))
                    protein_ids = batch.get('protein_ids', ['unknown'] * len(sample_ids))
                    self.logger.warning(f"验证批次 {batch_idx} 前向传播失败 - 跳过该批次")
                    self.logger.warning(f"错误类型: {type(e).__name__}")
                    self.logger.warning(f"错误信息: {str(e)}")
                    self.logger.warning(f"问题样本: {list(zip(sample_ids, protein_ids))}")
                    self.logger.warning(f"批次大小: {len(sample_ids)}")
                

                if self.is_distributed:
                    batch_failed_tensor = torch.tensor(int(batch_failed), device=self._device)
                    dist.all_reduce(batch_failed_tensor, op=dist.ReduceOp.MAX)
                    
                    # 多GPU训练，如果任何一个进程失败，所有进程都跳过这个批次
                    if batch_failed_tensor.item() > 0:
                        if not batch_failed and self.is_main_process:
                            self.logger.warning(f"验证批次 {batch_idx}: 其他进程失败，主进程同步跳过")
                        continue
                elif batch_failed:
                    # 单GPU训练直接跳过
                    continue
                

                for name, loss_val in losses.items():
                    if name not in total_losses:
                        total_losses[name] = 0.0
                    total_losses[name] += loss_val.item()       # total_losses字典的每个键对应一种损失，每个值是一个浮点数，该浮点数对sample level loss（即已经在一个batch内取平均值）累加
                

                logits = outputs['logits']  # [batch_size, seq_len, num_classes]
                labels = batch['labels']    # [batch_size, seq_len]
                
                if 'valid_mask' in outputs:
                    valid_mask = outputs['valid_mask']
                else:
                    raise RuntimeError('valid_mask not in outputs')
                

                probs = torch.softmax(logits, dim=-1)
                if self.model_config['classification_output_dim'] == 2:
                    preds = torch.argmax(probs, dim=-1)
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
                    # 多分类处理
                    preds = torch.argmax(probs, dim=-1)
                    num_classes = self.model_config['classification_output_dim']
                    
                    # 1 计算二分类指标（活性位点vs非活性位点），计算方式与EasIFA一致
                    binary_labels = (labels != 0).float()
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
                    

                    # 2 计算多分类指标，计算方式与EasIFA不一致
                    for site_type in range(1, num_classes):                 # 跳过背景类(class 0)
                        binary_labels = (labels == site_type).float()       # [batch_size, seq_len]，标记出哪些位置的真实类别等于当前的 site_type
                        binary_preds = (preds == site_type).float()         # [batch_size, seq_len]，标记哪些位置被预测为当前的 site_type。
                        site_prob = probs[:, :, site_type]                  # [batch_size, seq_len]，从模型输出的概率分布 probs 中，抽取当前类别 site_type 的预测概率。
                        
                        # 如果当前批次中不存在此类，则跳过
                        if binary_labels.sum() == 0:
                            continue
                        
                        site_metrics = compute_binary_classification_metrics(
                            predictions=binary_preds,
                            probabilities=site_prob,
                            labels=binary_labels,
                            valid_mask=valid_mask,
                            metrics=['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc', 'ap', 'mcc']
                        )
                        
                        for metric_name, metric_values in site_metrics.items():
                            all_metrics[f"pre_type_{site_type}_{metric_name}"].extend(metric_values)
                    

                    # 3 计算多分类指标，计算方式与EasIFA一致
                    multi_metrics = compute_multiple_classification_metrics(
                        pred=preds,
                        gt=labels,
                        num_site_types=num_classes,
                        valid_mask=valid_mask
                    )
                    
                    for metric_name, values in multi_metrics.items():
                        if values:  # 确保有值
                            # 将metric_name中的连字符替换为下划线，保持命名一致性
                            metric_key = metric_name.replace('-', '_')
                            all_metrics[metric_key].extend(values)
        finally:
            # Only close progress bar if it's a tqdm object (main process)
            if hasattr(pbar, 'close'):
                pbar.close()
        
        # 计算平均损失, epoch_losses是字典，每个键对应一种损失，每个值是一个浮点数，该浮点数是sample level loss(先在一个batch内求平均值，然后对target_batches个'batch平均值'累加，然后再除以target_batches求平均)
        if target_batches > 0 and total_losses:
            epoch_losses = {name: loss / target_batches for name, loss in total_losses.items()}
        else:
            epoch_losses = {}
        
        # Synchronize validation losses across GPUs if using distributed training
        if self.is_distributed and epoch_losses:
            # Convert losses to tensors for synchronization
            loss_tensors = {}
            for name, loss in epoch_losses.items():
                loss_tensor = torch.tensor(loss, device=self._device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensors[name] = loss_tensor.item() / self.world_size
            epoch_losses = loss_tensors
        
        # 计算所有批次的平均指标
        classification_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:  # 确保有值
                classification_metrics[metric_name] = sum(values) / len(values)
        
        # Synchronize classification metrics across GPUs if using distributed training
        if self.is_distributed and classification_metrics:
            metric_tensors = {}
            for name, metric in classification_metrics.items():
                metric_tensor = torch.tensor(metric, device=self._device)
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                metric_tensors[name] = metric_tensor.item() / self.world_size
            classification_metrics = metric_tensors
        
        # 计算多分类宏平均指标
        if self.model_config['classification_output_dim'] > 2:
            # 寻找所有类特定指标
            class_metrics = defaultdict(list)
            for metric_name in list(classification_metrics.keys()):
                if metric_name.startswith('type_') and '_' in metric_name[5:]:
                    parts = metric_name.split('_')
                    if len(parts) >= 3:
                        base_metric = '_'.join(parts[2:])
                        class_metrics[base_metric].append(classification_metrics[metric_name])
            
            # 计算宏平均值
            for metric_name, values in class_metrics.items():
                if values:
                    classification_metrics[f"macro_{metric_name}"] = sum(values) / len(values)
        
        return epoch_losses, classification_metrics



class EvoSiteCombinedTrainer(EvoSiteTrainer):
    """
    Refactored trainer for the EvoSiteCombined model. It overrides the
    _setup_model method to ensure the correct model class is instantiated.
    """
    
    def __init__(self, config: Dict[str, Any], local_rank: int = -1):
        super().__init__(config, local_rank)



    def _setup_model(self):

        if self.is_main_process:
            self.logger.info("Initializing refactored EvoSiteCombined model (distributed-safe)...")

        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        self.model = EvoSiteCombined(**model_config)
        self.model.to(self._device)

        if hasattr(torch, 'compile') and self.train_config.get('use_torch_compile', False):
            try:
                compile_mode = self.train_config.get('torch_compile_mode', 'reduce-overhead')
                self.model = torch.compile(self.model, mode=compile_mode)
                if self.is_main_process:
                    self.logger.info(f"torch.compile enabled with {compile_mode} mode")
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"torch.compile failed: {e}")
        
        # The rest of the setup (DDP wrapping, logging) is the same as the base class.
        if self.is_distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
        
        if self.is_main_process:
            model_for_params = self.model.module if self.is_distributed else self.model
            total_params = sum(p.numel() for p in model_for_params.parameters())
            trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
            
            self.logger.info(f"EvoSiteCombined initialized with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
            config_path = os.path.join(self.train_config['output_dir'], 'model_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.model_config, f, default_flow_style=False)



class EvoSiteBetaTrainer(EvoSiteTrainer):

    def __init__(self, config: Dict[str, Any], local_rank: int = -1):
        # The loss function will be assigned in _setup_loss_functions
        self.loss_fn = None
        # Call the parent constructor to set up the basic training infrastructure.
        super().__init__(config, local_rank)


    def _update_dca_scaling_scheduler(self):
        """
        Override to skip DCA scaling scheduler setup for EvoSiteBeta.
        EvoSiteBeta uses DCA features within ConditionNet but does not perform
        the attention biasing that requires dynamic scaling.
        """
        if self.is_main_process:
            self.logger.info("EvoSiteBeta model detected - skipping DCA scaling scheduler setup.")
        pass  # No-op for EvoSiteBeta




    def _setup_model(self):
        if self.is_main_process:
            self.logger.info("Initializing EvoSiteBeta model...")
        
        # Add device to model config before instantiation.
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        self.model = EvoSiteBeta(**model_config)
        self.model = self.model.to(self._device)

        # =========================================== Top@unfreeze ESM2-650M layers ======================================================
        # 从配置中获取要解冻的层列表，如果未定义则默认为空列表 (保证向后兼容)
        unfreeze_layers_config = self.train_config.get('unfreeze_esm_layers', [])
        if unfreeze_layers_config:
            if self.is_main_process:
                self.logger.info(f"Unfreezing ESM layers based on config: {unfreeze_layers_config}")

            # 定位到真正的ESM模块 (nn.Module)
            # 路径: self.model (EvoSiteBeta) -> self.model.esm_model (EvolutionaryScaleModelingBeta) -> self.model.esm_model.model (ESM2)
            esm_module_to_unfreeze = self.model.esm_model.model
            
            # Case 1: 解冻所有层
            if unfreeze_layers_config == "all":
                for param in esm_module_to_unfreeze.parameters():
                    param.requires_grad = True
                if self.is_main_process:
                    self.logger.info("All ESM model parameters have been unfrozen.")
            # Case 2: 解冻指定层
            elif isinstance(unfreeze_layers_config, list) and len(unfreeze_layers_config) > 0:
                unfrozen_params_count = 0
                
                # 直接通过模块访问来解冻指定的Transformer层
                for layer_idx in unfreeze_layers_config:
                    # 配置是1-based, ModuleList是0-based
                    if 0 <= layer_idx - 1 < len(esm_module_to_unfreeze.layers):
                        target_layer = esm_module_to_unfreeze.layers[layer_idx - 1]
                        for param in target_layer.parameters():
                            param.requires_grad = True
                            unfrozen_params_count += 1
                    else:
                        if self.is_main_process:
                            self.logger.warning(f"Layer index {layer_idx} is out of range for ESM model. Skipping.")
                
                # 解冻最终的LayerNorm层，这是一个好习惯
                for param in esm_module_to_unfreeze.emb_layer_norm_after.parameters():
                    param.requires_grad = True
                    unfrozen_params_count += param.numel() # param.numel()更准确地计数参数量
                
                if self.is_main_process:
                    trainable_params_in_esm = sum(p.numel() for p in esm_module_to_unfreeze.parameters() if p.requires_grad)
                    self.logger.info(f"Unfrozen {trainable_params_in_esm:,} parameters in specified ESM layers and final LayerNorm.")
        else:
            if self.is_main_process:
                self.logger.info("ESM model remains fully frozen (default behavior).")
        # =========================================== Bottom@unfreeze ESM2-650M layers =================================================== 


        if hasattr(torch, 'compile') and self.train_config.get('use_torch_compile', False):
            try:
                compile_mode = self.train_config.get('torch_compile_mode', 'reduce-overhead')
                self.model = torch.compile(self.model, mode=compile_mode)
                if self.is_main_process:
                    self.logger.info(f"torch.compile enabled: {compile_mode} mode")
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"torch.compile failed: {e}")

        if self.is_distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True  # Important for models with conditional branches like EvoSiteBeta.
            )
            
        if self.is_main_process:
            model_for_params = self.model.module if self.is_distributed else self.model
            total_params = sum(p.numel() for p in model_for_params.parameters())
            trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
            
            self.logger.info(f"EvoSiteBeta initialized with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            if self.is_distributed:
                self.logger.info(f"Using DistributedDataParallel with {self.world_size} GPUs")
            
            config_path = os.path.join(self.train_config['output_dir'], 'model_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.model_config, f, default_flow_style=False)



    def _setup_data(self):
        # Log the start of the data setup process.
        self.logger.info("Setting up datasets for EvoSiteBeta...")
        
        self.train_dataset = EnzymeMSAStructureDataset(
            csv_file=self.train_config['train_data'],
            structure_path=self.train_config['structure_path'],
            msa_dir=self.train_config['msa_dir'],
            dca_dir=self.train_config['dca_dir'],
            dssp_exec_path=self.train_config.get('dssp_exec_path'),
            dssp_dir=self.train_config.get('dssp_dir'),
            graphec_radius=self.train_config.get('graphec_radius'),
            protein_max_length=self.train_config['max_seq_length'],
            max_msa_sequences=self.train_config['max_msa_sequences'],
            split="train",
            num_samples=self.train_config.get('train_num_samples', None),
            esm_model_name=self.train_config['esm_model_name'],
            msa_model_name=self.train_config['msa_model_name'],
            num_workers=self.train_config['num_workers'],
            preprocessing=self.train_config.get('preprocessing', None),
            preprocessing_dir=self.train_config.get('train_preprocessing_dir', None)
        )
        
        self.val_dataset = EnzymeMSAStructureDataset(
            csv_file=self.train_config['val_data'],
            structure_path=self.train_config['structure_path'],
            msa_dir=self.train_config['msa_dir'],
            dca_dir=self.train_config['dca_dir'],
            dssp_exec_path=self.train_config.get('dssp_exec_path'),
            dssp_dir=self.train_config.get('dssp_dir'),
            graphec_radius=self.train_config.get('graphec_radius'),
            protein_max_length=self.train_config['max_seq_length'],
            max_msa_sequences=self.train_config['max_msa_sequences'],
            split="val",
            num_samples=self.train_config.get('evaluate_num_samples', None),
            esm_model_name=self.train_config['esm_model_name'],
            msa_model_name=self.train_config['msa_model_name'],
            num_workers=self.train_config['num_workers'],
            preprocessing=self.train_config.get('preprocessing', None),
            preprocessing_dir=self.train_config.get('valid_preprocessing_dir', None)
        )
        
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False
            )
        else:
            self.train_sampler = None
            self.val_sampler = None

        if self.is_distributed:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_config['batch_size'],
                sampler=self.train_sampler,
                num_workers=self.train_config['num_workers'],
                collate_fn=collate_enzyme_batch,
                pin_memory=self.train_config['pin_memory'],
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.train_config['batch_size'],
                sampler=self.val_sampler,
                num_workers=self.train_config['num_workers'],
                collate_fn=collate_enzyme_batch,
                pin_memory=self.train_config['pin_memory'],
                drop_last=False
            )
        else:
            # For single GPU training, shuffle is controlled directly in the DataLoader.
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_config['batch_size'],
                shuffle=True,
                num_workers=self.train_config['num_workers'],
                collate_fn=collate_enzyme_batch,
                pin_memory=self.train_config['pin_memory'],
                prefetch_factor=4,
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.train_config['batch_size'],
                shuffle=False,
                num_workers=self.train_config['num_workers'],
                collate_fn=collate_enzyme_batch,
                pin_memory=self.train_config['pin_memory'],
                prefetch_factor=4,
                drop_last=False
            )
        
        # Log detailed dataset and dataloader information on the main process.
        if self.is_main_process:
            self.logger.info(f"Training samples: {len(self.train_dataset)}")
            self.logger.info(f"Validation samples: {len(self.val_dataset)}")
            self.logger.info(f"Training batches per GPU: {len(self.train_loader)}")
            self.logger.info(f"Validation batches per GPU: {len(self.val_loader)}")
            if self.is_distributed:
                self.logger.info(f"Total training batches across {self.world_size} GPUs: {len(self.train_loader) * self.world_size}")
                self.logger.info(f"Total validation batches across {self.world_size} GPUs: {len(self.val_loader) * self.world_size}")



    def _setup_loss_functions(self):
        """Setup the primary classification loss function for EvoSiteBeta."""
        self.loss_fn = create_loss_function(
            loss_type=self.train_config.get('loss_type', 'ce'),
            class_weights=self.train_config.get('class_weights', None),
            label_smoothing=self.train_config.get('label_smoothing', 0.0)
        ).to(self._device)

        if self.is_main_process:
            self.logger.info(f"Loss function initialized: {self.train_config.get('loss_type', 'ce')}")



    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for the EvoSiteBeta model. 
        
        Args:
            outputs: Dictionary from the model's forward pass, containing 'logits' and 'protein_mask'.
            batch: The input batch dictionary, containing 'targets'.
            
        Returns:
            A dictionary containing 'main_loss' and 'total_loss'.
        """
        losses = {}
        
        # 'logits' are features for valid residues only, shape: (|V_res|, num_classes).
        # 'targets' contains corresponding labels for these residues, shape: (|V_res|,).
        logits = outputs['logits']
        labels = batch['targets']

        assert logits.size(0) == labels.size(0), f"Mismatch between number of residues in logits ({logits.size(0)}) and targets ({labels.size(0)}). "
        
        # Ensure labels are of the correct type for the loss function.
        labels = labels.long()

        # The main classification loss is computed on the unpacked, valid residues.
        if logits.dtype == torch.float16:
            logits = logits.float()
        main_loss = self.loss_fn(logits, labels)
        
        # For EvoSiteBeta, the total loss is just the main classification loss.
        # No auxiliary or alignment losses are generated by this model architecture.
        losses['main_loss'] = main_loss
        losses['total_loss'] = main_loss
        
        return losses
    


    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch. This method overrides the base to adapt the forward pass
        to EvoSiteBeta's specific output format.
        """
        self.model.train()
        
        total_losses = defaultdict(float)
        num_batches = len(self.train_loader)
        accumulation_steps = self.train_config.get('accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.train_config['epochs']} [TRAIN]", unit="batch", disable=not self.is_main_process)
        
        for batch_idx, batch in enumerate(pbar):
            batch_failed = False
            try:
                if batch is None:
                    raise ValueError("Batch is None, likely due to a collate_fn error.")
                batch = self._move_batch_to_device(batch)

                with autocast(enabled=self.use_amp):
                    logits, protein_mask = self.model(batch)
                    outputs = {"logits": logits, "protein_mask": protein_mask}
                    
                    losses = self.compute_loss(outputs, batch)
                    loss = losses['total_loss'] / accumulation_steps

                    # Check for NaN/Inf values in loss
                    if not self._check_loss_validity(loss, f"batch_{batch_idx}"):
                        raise ValueError(f"Invalid loss detected: {loss}")
                    
            except Exception as e:
                batch_failed = True
                if self.is_main_process:
                    uniprot_ids = batch.get('uniprot_ids', ['N/A'] * self.train_config.get('batch_size', 1))
                    self.logger.warning(f"Training batch {batch_idx} failed and will be skipped. Error: {e}")
                    self.logger.warning(f"Problematic UniProt IDs might be in: {uniprot_ids}")
                sys.stdout.flush()
                sys.stderr.flush()

            # Synchronize failure status across all GPUs. If one fails, all skip.
            if self.is_distributed:
                failure_tensor = torch.tensor(int(batch_failed), device=self._device)
                dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX)
                if failure_tensor.item() > 0:
                    if not batch_failed and self.is_main_process:
                         self.logger.warning(f"Batch {batch_idx}: Skipping due to failure on another process.")
                    continue  # All processes skip this batch
            elif batch_failed:
                continue # Single-GPU case, just skip
                
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate loss values for logging.
            for name, value in losses.items():
                if name not in total_losses:
                    total_losses[name] = 0.0
                total_losses[name] += value.item()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1

            if self.is_main_process:
                pbar.set_postfix(loss=f"{losses['total_loss'].item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Return the average losses for the epoch.
        epoch_losses = {name: total_loss / num_batches for name, total_loss in total_losses.items()}
        return epoch_losses



    @torch.no_grad()
    def validate_epoch(
        self,
        evaluate_num_samples: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:

        self.model.eval()
        total_losses = defaultdict(float)
        all_metrics = defaultdict(list)

        # Determine the number of batches to evaluate.
        val_loader = self.val_loader
        if evaluate_num_samples is not None:
            batch_size = self.train_config['batch_size']
            target_batches = (evaluate_num_samples + batch_size - 1) // batch_size
            target_batches = min(target_batches, len(val_loader))
        else:
            target_batches = len(val_loader)

        # Create progress bar only for main process
        if self.is_main_process:
            pbar = tqdm(
                enumerate(val_loader),
                total=target_batches,
                desc=f"Epoch {self.current_epoch + 1}/{self.train_config['epochs']} [VAL]",
                unit="batch"
            )
        else:
            pbar = enumerate(val_loader)
        
        try:
            for batch_idx, batch in pbar:
                if batch_idx >= target_batches:
                    break
                if batch is None:
                    continue

                batch_failed = False
                try:
                    batch = self._move_batch_to_device(batch)
                    with autocast(enabled=False):
                        flat_logits, protein_mask = self.model(batch)
                        outputs = {"logits": flat_logits, "protein_mask": protein_mask}
                        losses = self.compute_loss(outputs, batch)
                except Exception as e:
                    batch_failed = True
                    if self.is_main_process:
                        uniprot_ids = batch.get('uniprot_ids', ['N/A'])
                        self.logger.warning(f"Validation batch {batch_idx} failed and will be skipped. Error: {e}")
                        self.logger.warning(f"Problematic UniProt IDs might be in: {uniprot_ids}")

                if self.is_distributed:
                    failure_tensor = torch.tensor(int(batch_failed), device=self._device)
                    dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX)
                    if failure_tensor.item() > 0:
                        if not batch_failed and self.is_main_process:
                             self.logger.warning(f"Batch {batch_idx}: Skipping due to failure on another process.")
                        continue
                elif batch_failed:
                    continue
                
                for name, value in losses.items():
                    total_losses[name] += value.item()

                # =========================================== Top@Data Format Adaptation ====================================================== 
                flat_labels = batch['targets'].long()

                batch_size, max_len = protein_mask.shape
                labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=self._device)
                logits = torch.zeros(batch_size, max_len, self.model_config['classification_output_dim'], device=self._device)
                
                labels[protein_mask.bool()] = flat_labels
                logits[protein_mask.bool()] = flat_logits
                valid_mask = protein_mask
                # =========================================== Bottom@Data Format Adaptation =================================================== 

                probs = torch.softmax(logits, dim=-1)
                if self.model_config['classification_output_dim'] == 2:
                    preds = torch.argmax(probs, dim=-1)
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
                    # 多分类处理
                    preds = torch.argmax(probs, dim=-1)
                    num_classes = self.model_config['classification_output_dim']
                    
                    # 1 计算二分类指标（活性位点vs非活性位点），计算方式与EasIFA一致
                    binary_labels = (labels != 0).float()
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
                    

                    # 2 计算多分类指标，计算方式与EasIFA不一致
                    for site_type in range(1, num_classes):
                        binary_labels = (labels == site_type).float()
                        binary_preds = (preds == site_type).float()
                        site_prob = probs[:, :, site_type]
                        if binary_labels.sum() == 0:
                            continue
                                                
                        site_metrics = compute_binary_classification_metrics(
                            predictions=binary_preds,
                            probabilities=site_prob,
                            labels=binary_labels,
                            valid_mask=valid_mask,
                            metrics=['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc', 'ap', 'mcc']
                        )
                        
                        for metric_name, metric_values in site_metrics.items():
                            all_metrics[f"pre_type_{site_type}_{metric_name}"].extend(metric_values)
                    

                    # 3 计算多分类指标，计算方式与EasIFA一致
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
        finally:
            # Only close progress bar if it's a tqdm object (main process)
            if hasattr(pbar, 'close'):
                pbar.close()
        

        if target_batches > 0 and total_losses:
            epoch_losses = {name: loss / target_batches for name, loss in total_losses.items()}
        else:
            epoch_losses = {}
        
        # Synchronize validation losses across GPUs if using distributed training
        if self.is_distributed and epoch_losses:
            loss_tensors = {}
            for name, loss in epoch_losses.items():
                loss_tensor = torch.tensor(loss, device=self._device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensors[name] = loss_tensor.item() / self.world_size
            epoch_losses = loss_tensors
        
        # Calculate average for all collected metrics.
        classification_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                classification_metrics[metric_name] = sum(values) / len(values)
        
        # Synchronize classification metrics across GPUs if using distributed training
        if self.is_distributed and classification_metrics:
            metric_tensors = {}
            for name, metric in classification_metrics.items():
                metric_tensor = torch.tensor(metric, device=self._device)
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                metric_tensors[name] = metric_tensor.item() / self.world_size
            classification_metrics = metric_tensors
        
        # Calculate macro-average metrics from per-class metrics.
        if self.model_config['classification_output_dim'] > 2:
            class_metrics = defaultdict(list)
            for metric_name in list(classification_metrics.keys()):
                if metric_name.startswith('type_') and '_' in metric_name[5:]:
                    parts = metric_name.split('_')
                    if len(parts) >= 3:
                        base_metric = '_'.join(parts[2:])
                        class_metrics[base_metric].append(classification_metrics[metric_name])
            
            # Calculate the macro average.
            for metric_name, values in class_metrics.items():
                if values:
                    classification_metrics[f"macro_{metric_name}"] = sum(values) / len(values)

        return epoch_losses, classification_metrics



class EvoSiteGammaTrainer(EvoSiteBetaTrainer):
    """
    Trainer for the EvoSiteGamma model, which extends EvoSiteBeta with a multi-task
    learning framework. It computes the primary classification loss and supplementary
    auxiliary losses, all controlled by individual flags.
    """
    def __init__(self, config: Dict[str, Any], local_rank: int = -1):
        """
        Initializes the EvoSiteGammaTrainer, preparing for a flexible multi-task setup.
        """
        super().__init__(config, local_rank)

    def _setup_model(self):
        """
        Override to instantiate the EvoSiteGamma model with its flexible auxiliary tasks.
        """
        if self.is_main_process:
            self.logger.info("Initializing EvoSiteGamma model...")
        
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        self.model = EvoSiteGamma(**model_config)
        self.model = self.model.to(self._device)
        
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        
        if self.is_main_process:
            # Logging model parameters
            model_to_log = self.model.module if self.is_distributed else self.model
            total_params = sum(p.numel() for p in model_to_log.parameters())
            trainable_params = sum(p.numel() for p in model_to_log.parameters() if p.requires_grad)
            self.logger.info(f"EvoSiteGamma initialized with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def _setup_loss_functions(self):
        """
        Set up loss functions for the main task and all configurable auxiliary tasks.
        """
        # 1. Setup the main classification loss from the parent class
        super()._setup_loss_functions() # This initializes self.loss_fn

        # 2. Initialize flags and weights for all four auxiliary losses
        self.use_aux_coevolution_loss = self.model_config.get('use_aux_coevolution_head', False)
        self.use_aux_distance_loss = self.model_config.get('use_aux_distance_head', False)
        self.use_conservation_mlm_loss = self.model_config.get('use_conservation_weighted_mlm', False)
        self.use_coevolutionary_recon_loss = self.model_config.get('use_coevolutionary_reconstruction', False)

        self.aux_coevolution_weight = self.train_config.get('aux_coevolution_loss_weight', 0.2)
        self.aux_distance_weight = self.train_config.get('aux_distance_loss_weight', 0.5)
        self.conservation_mlm_weight = self.train_config.get('conservation_mlm_loss_weight', 0.3)
        self.coevolutionary_recon_weight = self.train_config.get('coevolutionary_recon_loss_weight', 0.3)

        # 3. Conditionally initialize loss functions for each task
        if self.use_aux_coevolution_loss:
            self.coevolution_loss_fn = nn.MSELoss()
        if self.use_aux_distance_loss:
            self.distance_loss_fn = nn.CrossEntropyLoss()
        
        # For both MLM losses, we need a base cross-entropy loss that doesn't reduce
        base_mlm_loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        if self.use_conservation_mlm_loss:
            self.conservation_mlm_loss_fn = ConservationWeightedMlmLoss(
                base_loss_fn=base_mlm_loss_fn,
                alpha=self.train_config.get('conservation_mlm_alpha', 1.0)
            )
        if self.use_coevolutionary_recon_loss:
            self.coevolution_recon_loss_fn = CoevolutionaryReconstructionLoss(
                base_loss_fn=nn.CrossEntropyLoss(), # This one can use reduction='mean'
                k=self.train_config.get('coevolutionary_recon_k', 3),
                mask_token_id=self.model_config.get('mlm_mask_token_id', 32)
            )

        if self.is_main_process:
            self.logger.info("--- Auxiliary Loss Configuration ---")
            self.logger.info(f"Co-evolution Prediction: Enabled={self.use_aux_coevolution_loss}, Weight={self.aux_coevolution_weight}")
            self.logger.info(f"Distance Prediction: Enabled={self.use_aux_distance_loss}, Weight={self.aux_distance_weight}")
            self.logger.info(f"Conservation Weighted MLM: Enabled={self.use_conservation_mlm_loss}, Weight={self.conservation_mlm_weight}")
            self.logger.info(f"Co-evolutionary Reconstruction: Enabled={self.use_coevolutionary_recon_loss}, Weight={self.coevolutionary_recon_weight}")
            self.logger.info("------------------------------------")


    def _prepare_mlm_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares a batch for any active MLM-based auxiliary tasks."""
        # This function should be idempotent; it modifies a copy.
        prepared_batch = batch.copy()
        
        # We need original token sequences to perform masking.
        # Since the dataloader provides a graph, we reconstruct the padded token tensor.
        # Note: This requires access to the alphabet, which we can get from the dataloader or model.
        # For simplicity, we assume access to the original tokens if they were collated.
        # If not, this part needs logic to extract and pad tokens from the graph.
        # The EvoSiteBeta trainer's dataloader doesn't provide 'sequences'.
        # This is a major dependency. Let's assume for now that if MLM is used,
        # the dataloader is adapted to provide 'sequences' and 'esm_valid_tokens_mask'.
        if 'sequences' not in batch:
            # If MLM is enabled but sequence data is missing, we cannot proceed.
            if self.use_conservation_mlm_loss or self.use_coevolutionary_recon_loss:
                 if self.is_main_process:
                    self.logger.warning("MLM tasks enabled, but 'sequences' tensor not found in batch. Skipping MLM preparation.")
            return prepared_batch # Return original batch
            
        input_tokens = batch['sequences'].clone()
        final_mask_for_loss = torch.zeros_like(input_tokens, dtype=torch.bool)
        
        # Co-evolutionary Reconstruction Masking (has priority)
        if self.use_coevolutionary_recon_loss:
            # This loss function has its own batch preparation logic
            prepared_batch = self.coevolution_recon_loss_fn.prepare_batch(prepared_batch)
            input_tokens = prepared_batch['coevo_masked_input_tokens']
            final_mask_for_loss |= prepared_batch['coevo_mask']

        # Standard MLM Masking for Conservation-Weighted Loss
        if self.use_conservation_mlm_loss:
            # Simple random masking logic
            valid_mask = batch['esm_valid_tokens_mask']
            # Probability of masking each token
            prob = 0.15
            # Decide which tokens to mask
            mask_decision = torch.rand_like(input_tokens.float()) < prob
            # Only mask valid tokens, and not tokens already masked by the other task
            mlm_mask = mask_decision & valid_mask & (~final_mask_for_loss)
            
            # Store targets and update input tokens
            prepared_batch['mlm_targets'] = input_tokens[mlm_mask]
            input_tokens[mlm_mask] = self.model_config.get('mlm_mask_token_id', 32)
            prepared_batch['mlm_mask'] = mlm_mask
            final_mask_for_loss |= mlm_mask

        # The final masked input for the model's MLM head
        prepared_batch['mlm_input_tokens'] = input_tokens
        return prepared_batch


    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch, with pre-processing for MLM tasks.
        """
        self.model.train()
        total_losses = defaultdict(float)
        num_batches = len(self.train_loader)
        accumulation_steps = self.train_config.get('accumulation_steps', 1)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.train_config['epochs']} [TRAIN]", unit="batch", disable=not self.is_main_process)
        
        for batch_idx, batch in enumerate(pbar):
            batch_failed = False
            try:
                if batch is None: raise ValueError("Batch is None.")
                batch = self._move_batch_to_device(batch)

                # Prepare batch for MLM tasks *before* the forward pass
                prepared_batch = self._prepare_mlm_batch(batch)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(prepared_batch)
                    losses = self.compute_loss(outputs, prepared_batch)
                    loss = losses['total_loss'] / accumulation_steps
                    if not self._check_loss_validity(loss, f"batch_{batch_idx}"):
                        raise ValueError(f"Invalid loss: {loss}")
            
            except Exception as e:
                # Exception handling logic... (same as parent)
                batch_failed = True
                if self.is_main_process:
                    uniprot_ids = batch.get('uniprot_ids', ['N/A'])
                    self.logger.warning(f"Training batch {batch_idx} failed: {e}. IDs: {uniprot_ids}")
                sys.stdout.flush()
                sys.stderr.flush()

            # Distributed failure synchronization (same as parent)
            if self.is_distributed:
                failure_tensor = torch.tensor(int(batch_failed), device=self._device)
                dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX)
                if failure_tensor.item() > 0:
                    continue
            elif batch_failed:
                continue

            # Backward pass and optimizer step (same as parent)
            if self.use_amp: self.scaler.scale(loss).backward()
            else: loss.backward()

            for name, value in losses.items(): total_losses[name] += value.item()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                if self.scheduler: self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            if self.is_main_process: pbar.set_postfix(loss=f"{losses['total_loss'].item():.4f}")
        
        return {name: total_loss / num_batches for name, total_loss in total_losses.items()}


    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute the combined multi-task loss for EvoSiteGamma based on enabled tasks.
        """
        losses = {}
        
        # 1. Main Classification Loss
        main_loss_dict = super().compute_loss(outputs, batch) # Re-use EvoSiteBeta's logic
        losses['main_loss'] = main_loss_dict['main_loss']
        total_loss = losses['main_loss']

        # --- Sequentially add weighted auxiliary losses if enabled ---
        
        # 2. Co-evolution Prediction Loss
        if self.use_aux_coevolution_loss and 'predicted_couplings' in outputs and 'coupling_matrices_valid' in batch:
            pred = outputs['predicted_couplings']
            true = batch['coupling_matrices_valid']
            if pred.shape == true.shape:
                loss = self.coevolution_loss_fn(pred, true)
                losses['aux_coevolution_loss'] = loss
                total_loss += self.aux_coevolution_weight * loss
            else:
                self.logger.warning(f"Shape mismatch for co-evolution loss. Pred: {pred.shape}, True: {true.shape}")

        # 3. Distance Prediction Loss
        if self.use_aux_distance_loss and 'predicted_distance_logits' in outputs and 'distance_bins_valid' in batch:
            pred_logits = outputs['predicted_distance_logits']
            true_bins = batch['distance_bins_valid']
            if pred_logits.shape[:2] == true_bins.shape:
                loss = self.distance_loss_fn(pred_logits.flatten(0, 1), true_bins.flatten().long())
                losses['aux_distance_loss'] = loss
                total_loss += self.aux_distance_weight * loss
            else:
                 self.logger.warning(f"Shape mismatch for distance loss. Pred: {pred_logits.shape}, True: {true_bins.shape}")

        # 4. Conservation-Weighted MLM Loss
        if self.use_conservation_mlm_loss and 'lm_head_logits' in outputs and 'mlm_mask' in batch:
            mlm_mask = batch['mlm_mask']
            if mlm_mask.any():
                logits = outputs['lm_head_logits'][mlm_mask]
                targets = batch['mlm_targets']
                # We need conservation scores for the masked tokens
                conservation = batch['conservation_scores'][mlm_mask]
                loss = self.conservation_mlm_loss_fn(logits, targets, conservation)
                losses['conservation_mlm_loss'] = loss
                total_loss += self.conservation_mlm_weight * loss

        # 5. Co-evolutionary Reconstruction Loss
        if self.use_coevolutionary_recon_loss and 'lm_head_logits' in outputs and 'coevo_mask' in batch:
            if batch['coevo_mask'].any():
                loss = self.coevolution_recon_loss_fn(outputs, batch)
                losses['coevolutionary_recon_loss'] = loss
                total_loss += self.coevolutionary_recon_weight * loss

        losses['total_loss'] = total_loss
        return losses
    

    
class EvoSitePhiGnetTrainer(EvoSiteBetaTrainer):
    def __init__(self, config: Dict[str, Any], local_rank: int = -1):
        """
        Initializes the EvoSitePhiGnetTrainer.

        Args:
            config (Dict[str, Any]): The full training configuration.
            local_rank (int): The local rank for distributed training.
        """
        super(EvoSiteBetaTrainer, self).__init__(config, local_rank)



    def _setup_model(self):

        if self.is_main_process:
            self.logger.info("Initializing EvoSitePhiGnet model...")
        
        # Add device to model config before instantiation.
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        self.model = EvoSitePhiGnet(**model_config)
        self.model = self.model.to(self._device)

        # =========================================== Top@unfreeze ESM2-650M layers ======================================================
        # 从配置中获取要解冻的层列表，如果未定义则默认为空列表 (保证向后兼容)
        unfreeze_layers_config = self.train_config.get('unfreeze_esm_layers', [])
        if unfreeze_layers_config:
            if self.is_main_process:
                self.logger.info(f"Unfreezing ESM layers based on config: {unfreeze_layers_config}")

            # 定位到真正的ESM模块 (nn.Module)
            # 路径: self.model (EvoSiteBeta) -> self.model.esm_model (EvolutionaryScaleModelingBeta) -> self.model.esm_model.model (ESM2)
            esm_module_to_unfreeze = self.model.esm_model.model
            
            # Case 1: 解冻所有层
            if unfreeze_layers_config == "all":
                for param in esm_module_to_unfreeze.parameters():
                    param.requires_grad = True
                if self.is_main_process:
                    self.logger.info("All ESM model parameters have been unfrozen.")
            # Case 2: 解冻指定层
            elif isinstance(unfreeze_layers_config, list) and len(unfreeze_layers_config) > 0:
                unfrozen_params_count = 0
                
                # 直接通过模块访问来解冻指定的Transformer层
                for layer_idx in unfreeze_layers_config:
                    # 配置是1-based, ModuleList是0-based
                    if 0 <= layer_idx - 1 < len(esm_module_to_unfreeze.layers):
                        target_layer = esm_module_to_unfreeze.layers[layer_idx - 1]
                        for param in target_layer.parameters():
                            param.requires_grad = True
                            unfrozen_params_count += 1
                    else:
                        if self.is_main_process:
                            self.logger.warning(f"Layer index {layer_idx} is out of range for ESM model. Skipping.")
                
                # 解冻最终的LayerNorm层，这是一个好习惯
                for param in esm_module_to_unfreeze.emb_layer_norm_after.parameters():
                    param.requires_grad = True
                    unfrozen_params_count += param.numel() # param.numel()更准确地计数参数量
                
                if self.is_main_process:
                    trainable_params_in_esm = sum(p.numel() for p in esm_module_to_unfreeze.parameters() if p.requires_grad)
                    self.logger.info(f"Unfrozen {trainable_params_in_esm:,} parameters in specified ESM layers and final LayerNorm.")
        else:
            if self.is_main_process:
                self.logger.info("ESM model remains fully frozen (default behavior).")
        # =========================================== Bottom@unfreeze ESM2-650M layers =================================================== 

        
        if hasattr(torch, 'compile') and self.train_config.get('use_torch_compile', False):
            try:
                compile_mode = self.train_config.get('torch_compile_mode', 'reduce-overhead')
                self.model = torch.compile(self.model, mode=compile_mode)
                if self.is_main_process:
                    self.logger.info(f"torch.compile enabled: {compile_mode} mode")
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"torch.compile failed: {e}")

        if self.is_distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False # EvoSitePhiGnet has a static graph, no unused params expected.
            )
            
        if self.is_main_process:
            model_for_params = self.model.module if self.is_distributed else self.model
            total_params = sum(p.numel() for p in model_for_params.parameters())
            trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
            
            self.logger.info(f"EvoSitePhiGnet initialized with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            if self.is_distributed:
                self.logger.info(f"Using DistributedDataParallel with {self.world_size} GPUs")
            
            config_path = os.path.join(self.train_config['output_dir'], 'model_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.model_config, f, default_flow_style=False)



class EvoSiteBetaPhiGnetTrainer(EvoSiteBetaTrainer):
    def __init__(self, config: Dict[str, Any], local_rank: int = -1):
        super(EvoSiteBetaTrainer, self).__init__(config, local_rank)



    def _setup_model(self):
        if self.is_main_process:
            self.logger.info("Initializing EvoSiteBetaPhiGnet model...")
        
        # Add device to model config before instantiation.
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        # Instantiate the correct model.
        self.model = EvoSiteBetaPhiGnet(**model_config)
        self.model = self.model.to(self._device)

        # Optional: PyTorch 2.0 compilation for performance.
        if hasattr(torch, 'compile') and self.train_config.get('use_torch_compile', False):
            try:
                compile_mode = self.train_config.get('torch_compile_mode', 'reduce-overhead')
                self.model = torch.compile(self.model, mode=compile_mode)
                if self.is_main_process:
                    self.logger.info(f"torch.compile enabled: {compile_mode} mode")
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"torch.compile failed: {e}")

        # Wrap with DDP if in a distributed environment.
        if self.is_distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True # Recommended for models with conditional logic
            )
            
        # Log model parameters on the main process.
        if self.is_main_process:
            model_for_params = self.model.module if self.is_distributed else self.model
            total_params = sum(p.numel() for p in model_for_params.parameters())
            trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
            
            self.logger.info(f"EvoSiteBetaPhiGnet initialized with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
            # Save the model config for reproducibility.
            config_path = os.path.join(self.train_config['output_dir'], 'model_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.model_config, f, default_flow_style=False)



class EvoSiteGraphECTrainer(EvoSiteBetaTrainer):
    def __init__(self, config: Dict[str, Any], local_rank: int = -1):
        super().__init__(config, local_rank)



    def _setup_model(self):
        if self.is_main_process:
            self.logger.info("Initializing EvoSiteGraphEC model...")
        
        # Add device to model config before instantiation.
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        self.model = EvoSiteGraphEC(**model_config)
        self.model = self.model.to(self._device)

        # =========================================== Top@unfreeze ESM2-650M layers ======================================================
        # 从配置中获取要解冻的层列表，如果未定义则默认为空列表 (保证向后兼容)
        unfreeze_layers_config = self.train_config.get('unfreeze_esm_layers', [])
        if unfreeze_layers_config:
            if self.is_main_process:
                self.logger.info(f"Unfreezing ESM layers based on config: {unfreeze_layers_config}")

            # 定位到真正的ESM模块 (nn.Module)
            # 路径: self.model (EvoSiteBeta) -> self.model.esm_model (EvolutionaryScaleModelingBeta) -> self.model.esm_model.model (ESM2)
            esm_module_to_unfreeze = self.model.esm_model.model
            
            # Case 1: 解冻所有层
            if unfreeze_layers_config == "all":
                for param in esm_module_to_unfreeze.parameters():
                    param.requires_grad = True
                if self.is_main_process:
                    self.logger.info("All ESM model parameters have been unfrozen.")
            # Case 2: 解冻指定层
            elif isinstance(unfreeze_layers_config, list) and len(unfreeze_layers_config) > 0:
                unfrozen_params_count = 0
                
                # 直接通过模块访问来解冻指定的Transformer层
                for layer_idx in unfreeze_layers_config:
                    # 配置是1-based, ModuleList是0-based
                    if 0 <= layer_idx - 1 < len(esm_module_to_unfreeze.layers):
                        target_layer = esm_module_to_unfreeze.layers[layer_idx - 1]
                        for param in target_layer.parameters():
                            param.requires_grad = True
                            unfrozen_params_count += 1
                    else:
                        if self.is_main_process:
                            self.logger.warning(f"Layer index {layer_idx} is out of range for ESM model. Skipping.")
                
                # 解冻最终的LayerNorm层，这是一个好习惯
                for param in esm_module_to_unfreeze.emb_layer_norm_after.parameters():
                    param.requires_grad = True
                    unfrozen_params_count += param.numel() # param.numel()更准确地计数参数量
                
                if self.is_main_process:
                    trainable_params_in_esm = sum(p.numel() for p in esm_module_to_unfreeze.parameters() if p.requires_grad)
                    self.logger.info(f"Unfrozen {trainable_params_in_esm:,} parameters in specified ESM layers and final LayerNorm.")
        else:
            if self.is_main_process:
                self.logger.info("ESM model remains fully frozen (default behavior).")
        # =========================================== Bottom@unfreeze ESM2-650M layers =================================================== 


        if hasattr(torch, 'compile') and self.train_config.get('use_torch_compile', False):
            try:
                compile_mode = self.train_config.get('torch_compile_mode', 'reduce-overhead')
                self.model = torch.compile(self.model, mode=compile_mode)
                if self.is_main_process:
                    self.logger.info(f"torch.compile enabled: {compile_mode} mode")
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"torch.compile failed: {e}")

        if self.is_distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True # Recommended for models with conditional logic
            )
            
        if self.is_main_process:
            model_for_params = self.model.module if self.is_distributed else self.model
            total_params = sum(p.numel() for p in model_for_params.parameters())
            trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
            
            self.logger.info(f"EvoSiteGraphEC initialized with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
            config_path = os.path.join(self.train_config['output_dir'], 'model_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.model_config, f, default_flow_style=False)



class EvoSiteDeltaTrainer(EvoSiteBetaTrainer):
    def __init__(self, config: Dict[str, Any], local_rank: int = -1):
        super().__init__(config, local_rank)



    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler with detailed parameter logging."""
        
        # =========================================== Top@PARAMETER ANALYSIS ======================================================
        self.logger.info("Analyzing model parameters for EvoSiteDelta...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = total_params - trainable_count
        
        self.logger.info(f"OVERALL PARAMETER STATISTICS:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_count:,}")
        self.logger.info(f"  Frozen parameters: {frozen_count:,}")
        self.logger.info(f"  Trainable ratio: {trainable_count/total_params:.2%}")
        
        trainable_params_by_component = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                component = name.split('.')[0]
                if component not in trainable_params_by_component:
                    trainable_params_by_component[component] = []
                trainable_params_by_component[component].append(name)
        for component, param_names in sorted(trainable_params_by_component.items()):
            self.logger.info(f"  {component.upper()} - {len(param_names)} parameters:")
            for name in param_names:
                self.logger.info(f"    - {name}")
        # =========================================== Bottom@PARAMETER ANALYSIS ===================================================
        
        # Setup optimizer
        optimizer_type = self.train_config.get('optimizer', 'adamw').lower()
        lr = float(self.train_config['learning_rate'])
        weight_decay = float(self.train_config['weight_decay'])
        
        # Get reaction attention model learning rate ratio
        reaction_attention_model_lr_ratio = self.train_config.get('reaction_attention_model_lr_ratio', 0.1)
        
        # Check if the model has a reaction attention model
        model_instance = self.model.module if self.is_distributed else self.model
        has_reaction_model = hasattr(model_instance, 'rxn_attn_model') and model_instance.rxn_attn_model is not None
        
        if has_reaction_model:
            self.logger.info(f"Setting up differential learning rates:")
            self.logger.info(f"  Base learning rate: {lr}")
            self.logger.info(f"  Reaction attention model lr ratio: {reaction_attention_model_lr_ratio}")
            self.logger.info(f"  Reaction attention model lr: {lr * reaction_attention_model_lr_ratio}")
            
            # Create efficient parameter groups: aggregate by lr (uniform weight decay)
            regular_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "rxn_attn_model" not in n]
            reaction_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "rxn_attn_model" in n]
            
            params_groups = []
            if regular_params:
                params_groups.append({"params": regular_params, "lr": lr})
            if reaction_params:
                params_groups.append({"params": reaction_params, "lr": lr * reaction_attention_model_lr_ratio})
            
            # Log parameter group statistics
            self.logger.info(f"Parameter groups created (total groups: {len(params_groups)}):")
            self.logger.info(f"  Regular parameters: {len(regular_params)} tensors")
            self.logger.info(f"  Reaction parameters: {len(reaction_params)} tensors")
            
            # Create optimizer with parameter groups (remove redundant top-level lr, uniform weight decay)
            if optimizer_type == 'adamw':
                self.optimizer = optim.AdamW(
                    params_groups,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            elif optimizer_type == 'adam':
                self.optimizer = optim.Adam(
                    params_groups,
                    weight_decay=weight_decay
                )
            elif optimizer_type == 'sgd':
                self.optimizer = optim.SGD(
                    params_groups,
                    weight_decay=weight_decay,
                    momentum=0.9
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_type}")
                
        else:
            # No reaction model, use standard optimizer setup with grouping for consistency
            self.logger.info("No reaction attention model found, using standard optimizer setup")
            
            params_groups = [{"params": trainable_params, "lr": lr}]
            
            # Log
            self.logger.info(f"Parameter groups created (total groups: {len(params_groups)}):")
            self.logger.info(f"  Parameters: {len(trainable_params)} tensors")
            
            if optimizer_type == 'adamw':
                self.optimizer = optim.AdamW(
                    params_groups,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            elif optimizer_type == 'adam':
                self.optimizer = optim.Adam(
                    params_groups,
                    weight_decay=weight_decay
                )
            elif optimizer_type == 'sgd':
                self.optimizer = optim.SGD(
                    params_groups,
                    weight_decay=weight_decay,
                    momentum=0.9
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Setup scheduler (step-based for better control with gradient accumulation)
        scheduler_type = self.train_config.get('scheduler', 'cosine')
        if scheduler_type != 'none':
            # Calculate total training steps considering gradient accumulation
            steps_per_epoch = len(self.train_loader) // self.train_config.get('accumulation_steps', 1)
            total_training_steps = steps_per_epoch * self.train_config['epochs']
            warmup_steps = int(steps_per_epoch * self.train_config.get('warmup_epochs', 0))
            self.scheduler = get_step_based_scheduler(
                self.optimizer,
                scheduler_type=scheduler_type,
                total_steps=total_training_steps,
                warmup_steps=warmup_steps,
                min_lr=self.train_config.get('min_lr', 1e-6)
            )
            
            self.logger.info(f"  Steps per epoch: {steps_per_epoch}")
            self.logger.info(f"  Total training steps: {total_training_steps}")
            self.logger.info(f"  Warmup steps: {warmup_steps}")
        else:
            self.scheduler = None
        
        # Log final optimizer setup
        self.logger.info(f"\nOPTIMIZER SETUP:")
        self.logger.info(f"  Optimizer: {optimizer_type}")
        self.logger.info(f"  Base learning rate: {lr}")
        self.logger.info(f"  Weight decay: {weight_decay}")
        self.logger.info(f"  Scheduler: {scheduler_type}")
        self.logger.info(f"  Parameter groups: {len(params_groups)}")
        self.logger.info(f"  Total trainable parameters: {trainable_count:,}")



    def _setup_model(self):
        if self.is_main_process:
            self.logger.info("Initializing EvoSiteDelta model...")
        
        # Add device to model config before instantiation.
        model_config = self.model_config.copy()
        model_config['device'] = self._device
        
        self.model = EvoSiteDelta(**model_config)
        self.model = self.model.to(self._device)

        # =========================================== Top@unfreeze ESM2-650M layers ======================================================
        unfreeze_layers_config = self.train_config.get('unfreeze_esm_layers', [])
        if unfreeze_layers_config:
            if self.is_main_process:
                self.logger.info(f"Unfreezing ESM layers based on config: {unfreeze_layers_config}")

            esm_module_to_unfreeze = self.model.esm_model.model
            
            if unfreeze_layers_config == "all":
                for param in esm_module_to_unfreeze.parameters():
                    param.requires_grad = True
                if self.is_main_process:
                    self.logger.info("All ESM model parameters have been unfrozen.")
            elif isinstance(unfreeze_layers_config, list) and len(unfreeze_layers_config) > 0:
                for layer_idx in unfreeze_layers_config:
                    if 0 <= layer_idx - 1 < len(esm_module_to_unfreeze.layers):
                        target_layer = esm_module_to_unfreeze.layers[layer_idx - 1]
                        for param in target_layer.parameters():
                            param.requires_grad = True
                    else:
                        if self.is_main_process:
                            self.logger.warning(f"Layer index {layer_idx} is out of range. Skipping.")
                
                for param in esm_module_to_unfreeze.emb_layer_norm_after.parameters():
                    param.requires_grad = True
                
                if self.is_main_process:
                    trainable_params_in_esm = sum(p.numel() for p in esm_module_to_unfreeze.parameters() if p.requires_grad)
                    self.logger.info(f"Unfrozen {trainable_params_in_esm:,} parameters in specified ESM layers.")
        else:
            if self.is_main_process:
                self.logger.info("ESM model remains fully frozen (default behavior).")
        # =========================================== Bottom@unfreeze ESM2-650M layers =================================================== 


        if hasattr(torch, 'compile') and self.train_config.get('use_torch_compile', False):
            try:
                compile_mode = self.train_config.get('torch_compile_mode', 'reduce-overhead')
                self.model = torch.compile(self.model, mode=compile_mode)
                if self.is_main_process:
                    self.logger.info(f"torch.compile enabled: {compile_mode} mode")
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"torch.compile failed: {e}")

        if self.is_distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
            
        if self.is_main_process:
            model_for_params = self.model.module if self.is_distributed else self.model
            total_params = sum(p.numel() for p in model_for_params.parameters())
            trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
            
            self.logger.info(f"EvoSiteDelta initialized with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
            config_path = os.path.join(self.train_config['output_dir'], 'model_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.model_config, f, default_flow_style=False)



class ECEmbeddingNetTrainer:
    """
    * ECEmbeddingNetTrainer 负责训练 EC 嵌入网络, 使用验证集上的两个指标来保存最佳模型：
        * EC注释的加权F1分数 (`best_model_ec_f1.pt`)
        * 下游位点预测任务的多分类MCC (`best_model_site_mcc.pt`)
    * 顺序问题
        无序
            * self.id_ec_train: 字典
            * self.ec_id_train: 字典
            * protein_features_dict: 字典
            * cluster_center_model: 字典
        有序
            * 应确保一致顺序组A：
                * self.esm_emb
                * self.ordered_uniprot_ids
                * model(self.esm_emb)得到的embeddings
            * 应确保一致顺序组B：
                * self.dist_map： 字典，有序。与sorted(self.ec_id_train.keys())顺序一致
                * ecs = sorted(cluster_center_model.keys())： 列表，有序，与sorted(self.ec_id_train.keys())顺序一致
                * MultiPosNeg_dataset_with_mine_EC.full_list: 列表，有序，与sorted(self.ec_id_train.keys())顺序一致
    * 理解dist map
        * dist_map 决定采样分布 → 采样分布决定训练批次 → 训练批次更新模型 → 新模型更新 dist_map。
        * dist_map随着模型更新动态变化，相当于模型对 EC 空间的“主观认知”。
    """
    def __init__(
            self, 
            config: dict, 
            model: nn.Module, 
            train_protein_features: Dict[str, torch.Tensor],
            valid_protein_features: Dict[str, torch.Tensor],
            train_residue_features: Dict[str, torch.Tensor],
            valid_residue_features: Dict[str, torch.Tensor],
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame
        ):
        """
        Args:
            config: 总配置文件config/ClusterSite.yaml
            model: ECEmbeddingNet模型实例
            train_protein_features_dict (Dict[str, torch.Tensor]): 训练集蛋白质特征字典（键之间无序）
                {
                    uniprot_id_1: protein_tensor,
                    uniprot_id_2: protein_tensor,
                }
            valid_protein_features_dict (Dict[str, torch.Tensor]): 验证集蛋白质特征字典（键之间无序）
            train_residue_features_dict (Dict[str, torch.Tensor]): 训练集残基特征字典
            valid_residue_features_dict (Dict[str, torch.Tensor]): 验证集残基特征字典
            train_df: 已经在 ClusterSiteTrainer 中预处理好的训练集 DataFrame
            valid_df: 已经在 ClusterSiteTrainer 中预处理好的验证集 DataFrame
        """
        self.config = config

        self.device = torch.device(self.config['common']['device'])
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32
        
        assert model is not None, "ECEmbeddingNetTrainer requires an injected ECEmbeddingNet model."
        self.model = model.to(self.device)
        self.model.float()

        self.optimizer = self.get_optimizer()
        self.scaler = GradScaler() if self.use_amp else None
        self.criterion = SupConHardLoss
        self.best_loss = float('inf')
        self.best_ec_f1 = 0.0
        self.best_site_mcc = -1.0

        # =========================================== Top@train set(Protein-level for EC) ====================================================== 
        self.train_protein_features_dict = train_protein_features
        self.id_ec_train, self.ec_id_train = self.get_ec_id_dict(train_df, available_protein_ids=set(self.train_protein_features_dict.keys()))
        self.ordered_uniprot_ids = []
        self.ec_to_indices_map = defaultdict(list)
        current_index = 0
        for ec in sorted(self.ec_id_train.keys()):
            for uid in sorted(self.ec_id_train[ec]):
                self.ordered_uniprot_ids.append(uid)
                self.ec_to_indices_map[ec].append(current_index)
                current_index += 1
        
        ordered_embeddings_list = []
        for uid in self.ordered_uniprot_ids:
            if uid in self.train_protein_features_dict:
                ordered_embeddings_list.append(self.train_protein_features_dict[uid])
            else:
                raise ValueError(f"UniProt ID '{uid}' are missing in the ESM embedding dictionary. Please check the data consistency.")
    
        self.esm_emb = torch.stack(ordered_embeddings_list).to(device=self.device, dtype=self.dtype)    # (N, D), 在后面的流程中，一定要确保self.emb的第一维度的顺序不变              # Map Uniprot ID to index in esm_emb
        self.dist_map = self.get_dist_map(model=None)
        self.train_loader = self.get_dataloader(self.dist_map, self.id_ec_train, self.ec_id_train)
        # =========================================== Bottom@train set(Protein-level for EC) =================================================== 


        # =========================================== Top@valid set(Protein-level for EC) ====================================================== 
        assert valid_protein_features is not None, "Validation protein features dictionary must be provided."
        assert valid_residue_features is not None, "Validation residue features dictionary must be provided."
        self.valid_protein_features_dict = valid_protein_features
        self.id_ec_valid, _ = self.get_ec_id_dict(valid_df, available_protein_ids=set(self.valid_protein_features_dict.keys()))
        # =========================================== Bottom@valid set(Protein-level for EC) =================================================== 


        # =========================================== Top@Data Pools (Residue-level for Site) ============================
        (self.train_site_id_ec, 
        _, 
        _, 
        self.train_site_labels, 
        self.train_site_flat_features, 
        self.train_site_ec_pools_indexed, 
        self.train_site_uids_ordered, 
        self.train_site_uid_to_offset
        ) = build_data_pools(dataframe=train_df, embedding_dict=train_residue_features, num_workers=self.config['common']['num_workers'], ec_level=self.config['common']['ec_level'])
        
        # $$ 将Residue-level for Site和Protein-level for EC的id_ec_valid对齐
        (self.valid_site_id_ec,
        _, 
        _, 
        self.valid_site_labels,
        self.valid_site_flat_features, 
        self.valid_site_ec_pools_indexed,
        self.valid_site_uids_ordered, 
        self.valid_site_uid_to_offset
        ) = build_data_pools(dataframe=valid_df, embedding_dict=valid_residue_features, num_workers=self.config['common']['num_workers'], ec_level=self.config['common']['ec_level'])
        # =========================================== Bottom@Data Pools (Residue-level for Site) ==========================



    def train(self):
        pbar = tqdm(range(1, self.config['train']['ECEmbeddingNetTrainer']['epoch'] + 1), desc="ECEmbeddingNet Training")
        for epoch in pbar:
            if epoch % self.config['train']['ECEmbeddingNetTrainer']['adaptive_rate'] == 0 and epoch != self.config['train']['ECEmbeddingNetTrainer']['epoch'] + 1:
                if self.config['train']['ECEmbeddingNetTrainer']['reset_optimizer']:
                    self.optimizer = self.get_optimizer()
                
                # save updated model
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

                # sample new distance map
                self.dist_map = self.get_dist_map(model=self.model)
                self.train_loader = self.get_dataloader(self.dist_map, self.id_ec_train, self.ec_id_train)

            train_loss = self.train_epoch(epoch)    # train for one epoch
            metrics = self.valid_epoch(epoch)       # valid for one epoch

            ec_f1 = metrics['ec_f1']
            site_mcc = metrics['site_mcc']
            pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'valid_ec_f1': f'{ec_f1:.4f}',
                'valid_site_mcc': f'{site_mcc:.4f}',
                'best_f1': f'{self.best_ec_f1:.4f}',
                'best_mcc': f'{self.best_site_mcc:.4f}'
            })

            # save the best model based on ec_f1 and site_mcc
            if ec_f1 > self.best_ec_f1:
                self.best_ec_f1 = ec_f1
                self.save_checkpoint('best_model_ec_f1.pt')
            if site_mcc > self.best_site_mcc:
                self.best_site_mcc = site_mcc
                self.save_checkpoint('best_model_site_mcc.pt')



    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.
        for batch, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp, dtype=self.dtype):
                model_emb = self.model(data.to(device=self.device, dtype=self.dtype))
                loss = self.criterion(model_emb, self.config['train']['ECEmbeddingNetTrainer']['temp'], self.config['train']['ECEmbeddingNetTrainer']['n_pos'])
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
        return total_loss/(batch + 1)
    


    @torch.no_grad()
    def valid_epoch(self, epoch):
        self.model.eval()
        metrics = {}

        # protein-level
        with autocast(enabled=self.use_amp, dtype=self.dtype):
            reference_protein_features = self.model(self.esm_emb.to(self.device, self.dtype))
            query_ids = sorted(self.valid_protein_features_dict.keys())
            query_protein_features_initial = torch.stack([self.valid_protein_features_dict[uid] for uid in query_ids])
            query_protein_features = self.model(query_protein_features_initial.to(self.device, self.dtype))
        
        # residue-level
        reference_residue_embs = encode_residue_feature(flat_features=self.train_site_flat_features, model=self.model, batch_size=6000000, device=self.device, use_amp=self.use_amp, dtype=self.dtype)      # 每增加2000000个残基，越增加6GB显存
        query_residue_embs = encode_residue_feature(flat_features=self.valid_site_flat_features, model=self.model, batch_size=6000000, device=self.device, use_amp=self.use_amp, dtype=self.dtype)

        # EC 注释评估
        ec_metrics, _ = evaluate_ec_prediction(
            reference_protein_features=reference_protein_features,
            query_protein_features=query_protein_features,
            ec_to_indices_map=self.ec_to_indices_map,
            id_ec_query=self.id_ec_valid,
            query_ids=query_ids,
            top_k=1, # CLEAN是10，但是目前的数据集是单标签场景，即每个酶只记录有一个EC号
            device=self.device
        )
        metrics.update(ec_metrics)

        # 位点预测评估
        site_metrics_dict, _ = evaluate_site_type_prediction(
            reference_residue_features=reference_residue_embs,
            query_residue_features=query_residue_embs,
            ec_site_pools_indexed_reference=self.train_site_ec_pools_indexed,
            id_ec_query=self.valid_site_id_ec,
            labels_dict_query=self.valid_site_labels,
            uid_to_offset_query=self.valid_site_uid_to_offset,
            device=self.device,
            prob_mode=self.config['train']['StackerCTrainer']['prob_mode'],
        )
        if 'multi-class mcc' in site_metrics_dict and site_metrics_dict['multi-class mcc']:
            metrics['site_mcc'] = np.mean(site_metrics_dict['multi-class mcc']) # 计算多分类mcc的平均值
        else:
            raise RuntimeError("Site multi-class MCC calculation failed during validation.")
        
        self.model.train()
        return metrics



    def get_ec_id_dict(self, df: pd.DataFrame, available_protein_ids: set) -> dict:
        """
        适配SwissProt E-RXN ASA数据集格式的EC字典构建函数
        
        Returns:
            id_ec: Dict mapping Uniprot ID to list of EC numbers
                   {'L0E4H0': ['1.1.1.104'], 'A0A000': ['2.3.4.5', '1.1.1.1']}
            
            ec_id_dict: Dict mapping EC number to list of Uniprot IDs (converted from set to list)
                {'1.1.1.104': ['L0E4H0'], '2.3.4.5': ['A0A000'], '1.1.1.1': ['A0A000']}
        """
        ec_level = self.config['common']['ec_level']
        get_ec = (lambda ec: '.'.join(str(ec).split('.')[:ec_level]) if ec_level < 4 else str(ec))

        filtered_df = df[df['Uniprot ID'].isin(available_protein_ids)]
    
        id_ec = {}
        ec_id = {}
        
        for idx, row in filtered_df.iterrows():
            uniprot_id = row['Uniprot ID']
            ec_list = row['valid_ec_list']  # 这是一个 list
            current_ecs = list(dict.fromkeys(get_ec(ec) for ec in ec_list))

            # 存储Uniprot ID到EC号的映射
            id_ec[uniprot_id] = current_ecs
            
            # 存储EC号到Uniprot ID的映射
            for ec in current_ecs:
                if ec not in ec_id:
                    ec_id[ec] = set()
                ec_id[ec].add(uniprot_id)
        ec_id_dict = {key: list(ec_id[key]) for key in ec_id.keys()}
        return id_ec, ec_id_dict


    def get_dataloader(self, dist_map, id_ec, ec_id):
        params = {
            'batch_size': self.config['train']['ECEmbeddingNetTrainer']['batch_size'],
            'shuffle': self.config['train']['ECEmbeddingNetTrainer']['shuffle'],
            'num_workers': self.config['train']['ECEmbeddingNetTrainer']['num_workers'],
            'pin_memory': True,
        }
        negative = self.mine_hard_negative(dist_map, 100)    # 挖掘硬负样本：为每个 EC 找到最近的 100 个不同 EC
        train_data = MultiPosNeg_dataset_with_mine_EC(  # 创建数据集：每个样本包含 1个anchor + 9个正样本 + 30个负样本
            id_ec=id_ec, 
            ec_id=ec_id, 
            mine_neg=negative,
            n_pos=self.config['train']['ECEmbeddingNetTrainer']['n_pos'], 
            n_neg=self.config['train']['ECEmbeddingNetTrainer']['n_neg'],
            protein_features_dict=self.train_protein_features_dict)
        train_loader = torch.utils.data.DataLoader(train_data, **params)
        return train_loader


    def get_optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['train']['ECEmbeddingNetTrainer']['learning_rate'], 
            betas=(0.9, 0.999)
        )


    def find_first_non_zero_distance(self, data):
        for index, (name, distance) in enumerate(data):
            if distance != 0:
                return index
        return None 


    def mine_hard_negative(self, dist_map, knn=10):
        """
        硬负样本挖掘函数
        
        参数:
            dist_map: 嵌套字典，存储所有 EC 编号之间的距离
                    结构: {'EC1': {'EC1': 0.0, 'EC2': 2.5, ...}, ...}
            knn: 为每个 EC 选择的最近邻负样本数量（默认10个）
        
        返回:
            negative: 字典，包含每个 EC 的硬负样本候选池及其采样权重
        """
        #print("The number of unique EC numbers: ", len(dist_map.keys()))
        ecs = sorted(list(dist_map.keys())) # 示例: ecs = ['1.1.1.1', '2.3.4.5', '3.2.1.14', ...]
        negative = {}
        # print("Mining hard negatives:")
        for target in tqdm(ecs, total=len(ecs), desc="Mining hard negatives", disable=True):
            # 排序结果示例:
            # sorted_orders = [
            #     ('1.1.1.1', 0.0),      # 自己，距离为 0
            #     ('1.1.1.2', 2.5),      # 最近的 EC
            #     ('1.1.1.3', 3.8),      # 第二近的 EC
            #     ('2.3.4.5', 15.2),     # 较远的 EC
            #     ...
            # ]
            sorted_orders = sorted(dist_map[target].items(), key=lambda x: x[1], reverse=False)
            assert sorted_orders != None, "all clusters have zero distances!"
            neg_ecs_start_index = self.find_first_non_zero_distance(sorted_orders)
            closest_negatives = sorted_orders[neg_ecs_start_index:neg_ecs_start_index + knn]
            freq = [1/i[1] for i in closest_negatives]
            neg_ecs = [i[0] for i in closest_negatives]        
            normalized_freq = [i/sum(freq) for i in freq]

            # 最终结构示例:
            # negative['1.1.1.1'] = {
            #     'weights': [0.571, 0.286, 0.143, ...],  # 10 个权重
            #     'negative': ['1.1.1.2', '1.1.1.3', '1.1.1.5', ...]  # 10 个 EC
            # }
            negative[target] = {
                'weights': normalized_freq,
                'negative': neg_ecs
            }
        return negative


    def get_dist_map(self, model=None, dot=False):
        '''
        Get the distance map for training, size of (N_EC_train, N_EC_train)
        between all possible pairs of EC cluster centers

        Args:
            model (nn.Module, optional): 
                - 如果提供模型，则使用模型对嵌入进行前向传播，得到更新后的嵌入，再计算距离图 (用于训练中更新)。
                - 如果为 None，则直接使用初始的 self.esm_emb 计算距离图 (用于初始化)。
            dot (bool): 是否使用点积距离。
        Returns:
            model_dist (dict): 嵌套字典，存储所有 EC 编号
                model_dist = {
                    'EC1': {
                        'EC1': distance_value_11,  # float
                        'EC2': distance_value_12,  # float
                        'EC3': distance_value_13,  # float
                        ...
                    },
                    'EC2': {
                        'EC1': distance_value_21,  # float
                        'EC2': distance_value_22,  # float
                        'EC3': distance_value_23,  # float
                        ...
                    },
                    ...
                }
        '''
        with torch.no_grad():
            # inference all queries at once to get model embedding
            if model is not None:
                model.eval()
                with autocast(enabled=self.use_amp, dtype=self.dtype):
                    model_emb = model(self.esm_emb.to(device=self.device, dtype=self.dtype))
            else:
                model_emb = self.esm_emb


            # calculate cluster center by averaging all embeddings in one EC
            """
            cluster_center_model = {
                '1.1.1.1': tensor([0.1, 0.2, ..., 0.3]),  # 128维
                '2.3.4.5': tensor([0.4, 0.5, ..., 0.6]),  # 128维
                ...
            }
            """
            cluster_center_model = {}
            for ec, indices in tqdm(self.ec_to_indices_map.items(), desc="Calculating Cluster Centers", disable=True):
                if not indices: continue 
                emb_cluster = model_emb[indices]
                cluster_center = emb_cluster.mean(dim=0)
                cluster_center_model[ec] = cluster_center.cpu()

            # organize cluster centers in a matrix
            total_ec_n, out_dim = len(self.ec_id_train.keys()), model_emb.size(1)
            model_lookup = torch.zeros(total_ec_n, out_dim, device=self.device, dtype=self.dtype)
            ecs = sorted(cluster_center_model.keys()) # 对 cluster_center_model.keys() 进行排序，确保 ecs 列表顺序恒定
            for i, ec in enumerate(ecs):
                model_lookup[i] = cluster_center_model[ec]
            model_lookup = model_lookup.to(device=self.device, dtype=self.dtype)
            # calculate pairwise distance map between total_ec_n * total_ec_n pairs
            print(f'Calculating distance map, number of unique EC is {total_ec_n}')
            if dot:
                model_dist = self.dist_map_helper_dot(ecs, model_lookup, ecs, model_lookup)
            else:
                model_dist = self.dist_map_helper(ecs, model_lookup)

        if model is not None:
            model.train()

        return model_dist
    


    def dist_map_helper_dot(self, keys1, lookup1, keys2, lookup2):
        dist = {}
        lookup1 = F.normalize(lookup1, dim=-1, p=2)
        lookup2 = F.normalize(lookup2, dim=-1, p=2)
        for i, key1 in tqdm(enumerate(keys1)):
            current = lookup1[i].unsqueeze(0)
            dist_norm = (current - lookup2).norm(dim=1, p=2)
            dist_norm = dist_norm**2
            #dist_norm = (current - lookup2).norm(dim=1, p=2)
            dist_norm = dist_norm.detach().cpu().numpy()
            dist[key1] = {}
            for j, key2 in enumerate(keys2):
                dist[key1][key2] = dist_norm[j]
        return dist



    def dist_map_helper(self, keys, lookup):
        dist_matrix = torch.cdist(lookup, lookup, p=2)
        dist_matrix_cpu = dist_matrix.detach().cpu().numpy()
        dist = {keys[i]: {keys[j]: dist_matrix_cpu[i, j] for j in range(len(keys))} for i in range(len(keys))}
        return dist


    def save_checkpoint(self, filename):
        model_save_dir = self.config['common']['ECEmbeddingNet_model_path']
        os.makedirs(model_save_dir, exist_ok=True)
        save_path = os.path.join(model_save_dir, filename)
        torch.save(self.model.state_dict(), save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
class ResidueEncoderAlphaTrainer:

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 传入ResidueEncoderAlphaTrainer的config是总配置文件
        """
        self.config = config

        os.makedirs(self.config['common']['ResidueEncoder_model_path'], exist_ok=True)
        self.device = self.config['common']['device']
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32
        self.scaler = GradScaler() if self.use_amp else None
        self.best_val_metric = 0.0
        self.current_epoch = 0

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_loss_functions()
        self._setup_optimizations()



    def _setup_model(self):
        print("Initializing ResidueEncoderAlpha model...")
        self.model = ResidueEncoderAlpha(self.config['model']['ResidueEncoderAlpha']).to(self.device)
        self.model.float()

        # =========================================== Top@unfreeze ESM2 layers ======================================================
        # 从配置中获取要解冻的层列表，如果未定义则默认为空列表 (保证向后兼容)
        unfreeze_layers_config = self.config['train']['ResidueEncoderAlphaTrainer']['unfreeze_esm_layers']
        if unfreeze_layers_config:
            print(f"Unfreezing ESM layers based on config: {unfreeze_layers_config}")

            # 定位到真正的ESM模块 (nn.Module)
            # 路径: self.model (EvoSiteBeta) -> self.model.esm_model (EvolutionaryScaleModelingBeta) -> self.model.esm_model.model (ESM2)
            esm_module_to_unfreeze = self.model.esm_model.model
            
            # Case 1: 解冻所有层
            if unfreeze_layers_config == "all":
                for param in esm_module_to_unfreeze.parameters():
                    param.requires_grad = True
                if self.is_main_process:
                    print("All ESM model parameters have been unfrozen.")
            # Case 2: 解冻指定层
            elif isinstance(unfreeze_layers_config, list) and len(unfreeze_layers_config) > 0:
                unfrozen_params_count = 0
                
                # 直接通过模块访问来解冻指定的Transformer层
                for layer_idx in unfreeze_layers_config:
                    # 配置是1-based, ModuleList是0-based
                    if 0 <= layer_idx - 1 < len(esm_module_to_unfreeze.layers):
                        target_layer = esm_module_to_unfreeze.layers[layer_idx - 1]
                        for param in target_layer.parameters():
                            param.requires_grad = True
                            unfrozen_params_count += 1
                    else:
                        print(f"Layer index {layer_idx} is out of range for ESM model. Skipping.")
                
                # 解冻最终的LayerNorm层，这是一个好习惯
                for param in esm_module_to_unfreeze.emb_layer_norm_after.parameters():
                    param.requires_grad = True
                    unfrozen_params_count += param.numel() # param.numel()更准确地计数参数量
                
                trainable_params_in_esm = sum(p.numel() for p in esm_module_to_unfreeze.parameters() if p.requires_grad)
                print(f"Unfrozen {trainable_params_in_esm:,} parameters in specified ESM layers and final LayerNorm.")
        else:
            print("ESM model remains fully frozen (default behavior).")
        # =========================================== Bottom@unfreeze ESM2 layers =================================================== 

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model initialized with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")



    def _setup_data(self):
        print("Setting up datasets for ResidueEncoderAlpha using EnzymeActiveSiteDataset3...")
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
            filter_incomplete_ec=False,
        )

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
            filter_incomplete_ec=False,
        )
        

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['train']['ResidueEncoderAlphaTrainer']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['ResidueEncoderAlphaTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=self.config['train']['ResidueEncoderAlphaTrainer']['pin_memory'],
            prefetch_factor=4,
            drop_last=False
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config['train']['ResidueEncoderAlphaTrainer']['batch_size'],
            shuffle=False,
            num_workers=self.config['train']['ResidueEncoderAlphaTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=self.config['train']['ResidueEncoderAlphaTrainer']['pin_memory'],
            prefetch_factor=4,
            drop_last=False
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.valid_dataset)}")
        print(f"Training batches per GPU: {len(self.train_loader)}")
        print(f"Validation batches per GPU: {len(self.valid_loader)}")



    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler with detailed parameter logging."""
        
        # =========================================== Top@PARAMETER ANALYSIS ======================================================
        print("Analyzing model parameters...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = total_params - trainable_count
        
        print(f"OVERALL PARAMETER STATISTICS:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_count:,}")
        print(f"  Frozen parameters: {frozen_count:,}")
        print(f"  Trainable ratio: {trainable_count/total_params:.2%}")
        
        trainable_params_by_component = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                component = name.split('.')[0]
                if component not in trainable_params_by_component:
                    trainable_params_by_component[component] = []
                trainable_params_by_component[component].append(name)
        for component, param_names in sorted(trainable_params_by_component.items()):
            print(f"  {component.upper()} - {len(param_names)} parameters:")
            for name in param_names:
                print(f"    - {name}")
        # =========================================== Bottom@PARAMETER ANALYSIS ===================================================
        

        # Setup optimizer
        optimizer_type = self.config['train']['ResidueEncoderAlphaTrainer']['optimizer']
        lr = float(self.config['train']['ResidueEncoderAlphaTrainer']['learning_rate'])
        weight_decay = float(self.config['train']['ResidueEncoderAlphaTrainer']['weight_decay'])
        
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        

        # Setup scheduler (step-based for better control with gradient accumulation)
        scheduler_type = self.config['train']['ResidueEncoderAlphaTrainer']['scheduler']
        if scheduler_type != 'none':
            # Calculate total training steps considering gradient accumulation
            steps_per_epoch = len(self.train_loader) // self.config['train']['ResidueEncoderAlphaTrainer']['accumulation_steps']
            total_training_steps = steps_per_epoch * self.config['train']['ResidueEncoderAlphaTrainer']['epochs']
            warmup_steps = int(steps_per_epoch * self.config['train']['ResidueEncoderAlphaTrainer']['warmup_epochs'])
            self.scheduler = get_step_based_scheduler(
                self.optimizer,
                scheduler_type=scheduler_type,
                total_steps=total_training_steps,
                warmup_steps=warmup_steps,
                min_lr=self.config['train']['ResidueEncoderAlphaTrainer']['min_lr']
            )
            
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Total training steps: {total_training_steps}")
            print(f"  Warmup steps: {warmup_steps}")
        else:
            self.scheduler = None
        
        # log
        print(f"\nOPTIMIZER SETUP:")
        print(f"  Optimizer: {optimizer_type}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Scheduler: {scheduler_type}")
        print(f"  Parameters in optimizer: {len(trainable_params)} tensors")
        print(f"  Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")



    def _setup_loss_functions(self):
        """Setup the primary classification loss function for EvoSiteBeta."""
        self.loss_fn = create_loss_function(
            loss_type=self.config['train']['ResidueEncoderAlphaTrainer']['loss_type'],
            class_weights=self.config['train']['ResidueEncoderAlphaTrainer']['class_weights'],
            label_smoothing=self.config['train']['ResidueEncoderAlphaTrainer']['label_smoothing']
        ).to(self.device)

        print(f"Loss function initialized: {self.config['train']['ResidueEncoderAlphaTrainer']['loss_type']}")



    def _setup_optimizations(self):
        print("Setting up training environment optimizations...")
        
        if torch.cuda.is_available():
            try:
                # cuDNN settings
                enable_cudnn_benchmark = self.config['train']['ResidueEncoderAlphaTrainer']['cudnn_benchmark']
                if enable_cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True
                
                # Attention backend settings (for PyTorch 2.x+)
                use_efficient_backends = self.config['train']['ResidueEncoderAlphaTrainer']['use_efficient_attention_backends']
                if use_efficient_backends:
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                else:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                
                # TF32 precision settings (for Ampere GPUs+)
                use_tf32 = self.config['train']['ResidueEncoderAlphaTrainer']['use_tf32']
                if use_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    
            except Exception as e:
                print(f"部分训练环境优化设置失败: {e}")
        
        # Memory management settings
        self.enable_memory_optimization = self.config['train']['ResidueEncoderAlphaTrainer']['enable_memory_optimization']
        self.memory_cleanup_frequency = self.config['train']['ResidueEncoderAlphaTrainer']['memory_cleanup_frequency']



    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Training for {self.config['train']['ResidueEncoderAlphaTrainer']['epochs']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['train']['ResidueEncoderAlphaTrainer']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
        
            # Validation 
            if (epoch + 1) % self.config['train']['ResidueEncoderAlphaTrainer']['val_frequency'] == 0:
                val_metrics = self.validate_epoch()
                print(f"[Epoch {epoch + 1}] Train - avg_loss: {train_losses['avg_train_loss']:.4f}, total_loss: {train_losses['total_train_loss']:.4f}")
                print(f"[Epoch {epoch + 1}] Val Metrics - {', '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])}")
                
                # Check for best model - use correct metric name based on num_classes
                metric_name = 'mcc' if self.config['model']['ResidueEncoderAlpha']['num_class'] == 2 else 'multi_class_mcc'
                current_metric = val_metrics[metric_name]
                is_best = current_metric > self.best_val_metric
                
                if is_best:
                    self.best_val_metric = current_metric
                    print(f"New best model! MCC: {current_metric:.4f}")

                # Save checkpoint
                if (epoch + 1) % self.config['train']['ResidueEncoderAlphaTrainer']['save_frequency'] == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
            else:
                print(f"[Epoch {epoch + 1}] Train - avg_loss: {train_losses['avg_train_loss']:.4f}, total_loss: {train_losses['total_train_loss']:.4f}")
            
        
        # Training completed
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation metric: {self.best_val_metric:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(filename="final_checkpoint.pt")




    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        accumulation_steps = self.config['train']['ResidueEncoderAlphaTrainer']['accumulation_steps']
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config['train']['ResidueEncoderAlphaTrainer']['epochs']} [TRAIN]", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            batch_failed = False
            try:
                if batch is None:   raise ValueError("Batch is None, likely due to a collate_fn error.")
                batch = self._move_batch_to_device(batch)
                with autocast(enabled=self.use_amp, dtype=self.dtype):
                    logits, protein_mask = self.model(batch)
                    labels = batch['targets'].long()
                    assert logits.size(0) == labels.size(0), f"Mismatch between number of residues in logits ({logits.size(0)}) and targets ({labels.size(0)}). "
                    main_loss = self.loss_fn(logits, labels)
                    loss = main_loss / accumulation_steps
                    if not self._check_loss_validity(loss, f"batch_{batch_idx}"):   raise ValueError(f"Invalid loss detected: {loss}")
                    
            except Exception as e:
                batch_failed = True
                uniprot_ids = batch.get('uniprot_ids', ['N/A'] * self.config['train']['ResidueEncoderAlphaTrainer']['batch_size'])
                print(f"Training batch {batch_idx} failed and will be skipped. Error: {e}")
                print(f"Problematic UniProt IDs might be in: {uniprot_ids}")
                sys.stdout.flush()
                sys.stderr.flush()

            if batch_failed:
                continue
            
            try:
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError as e:
                if "SoftmaxBackward" in str(e):
                    print("NaN in SoftmaxBackward, skip batch.")
                    continue
                if "returned nan values in its" in str(e):
                    print(f"[Warning] NaN detected in backward at batch {batch_idx}: {str(e)}")
                    self.optimizer.zero_grad()
                    if self.use_amp and self.scaler is not None:
                        # 给 GradScaler 一个机会调整 scale，减小后续溢出的概率
                        self.scaler.update()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    continue
                raise
            total_loss += main_loss.item()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['ResidueEncoderAlphaTrainer']['gradient_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['ResidueEncoderAlphaTrainer']['gradient_clip'])
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()


            pbar.set_postfix(loss=f"{main_loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Return the average loss for the epoch.
        avg_loss = total_loss / num_batches
        return {'avg_train_loss': avg_loss, 'total_train_loss': total_loss}



    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch，返回分类指标。"""
        self.model.eval()
        all_metrics = defaultdict(list)
        num_classes = self.config['model']['ResidueEncoderAlpha']['num_class']
        
        pbar = tqdm(
            enumerate(self.valid_loader),
            total=len(self.valid_loader),
            desc=f"Epoch {self.current_epoch + 1}/{self.config['train']['ResidueEncoderAlphaTrainer']['epochs']} [VAL]",
            unit="batch"
        )
        
        try:
            for batch_idx, batch in pbar:
                if batch_idx >= len(self.valid_loader):
                    break
                if batch is None:
                    continue

                try:
                    batch = self._move_batch_to_device(batch)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        flat_logits, protein_mask = self.model(batch)
                        flat_labels = batch['targets'].long()
                        assert flat_logits.size(0) == flat_labels.size(0), f"Mismatch: logits({flat_logits.size(0)}) vs targets({flat_labels.size(0)})"
                except Exception as e:
                    uniprot_ids = batch.get('uniprot_ids', ['N/A'])
                    print(f"Validation batch {batch_idx} failed. Error: {e}, UniProt IDs: {uniprot_ids}")
                    continue

                # Data Format Adaptation: 将flat格式转换为batch格式
                batch_size, max_len = protein_mask.shape
                labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
                logits = torch.zeros(batch_size, max_len, num_classes, device=self.device)
                labels[protein_mask.bool()] = flat_labels
                logits[protein_mask.bool()] = flat_logits
                valid_mask = protein_mask

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

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
        finally:
            if hasattr(pbar, 'close'):
                pbar.close()

        # 计算所有指标的平均值
        classification_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                classification_metrics[metric_name] = sum(values) / len(values)
        for metric_name, value in sorted(classification_metrics.items()):
            print(f"  {metric_name}: {value:.4f}")
        return classification_metrics



    def _check_loss_validity(self, loss, step_name: str = "") -> bool:
        """
        简单检查 loss 是否为有效数值。
        返回:
            True  : loss 正常
            False : loss 为 NaN 或 Inf
        """
        # 注意：loss 这里应该是标量 tensor
        if torch.isnan(loss) or torch.isinf(loss):
            msg = f"{step_name}: Invalid loss detected {loss}"
            print(msg)
            return False

        return True
    


    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model checkpoint
            filename: Optional checkpoint filename (defaults to epoch-based naming)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pt"      # Use 1-based epoch numbering for checkpoint filename
        filepath = os.path.join(self.config['common']['ResidueEncoder_model_path'], filename)

        model_to_save = self.model

        
        checkpoint = {
            # Model state: core model parameters
            'model_state': {
                'state_dict': model_to_save.state_dict(),
                'model_class': model_to_save.__class__.__name__
            },
            
            # Optimizer state: optimization parameters and momentum
            'optimizer_state': {
                'state_dict': self.optimizer.state_dict(),
                'optimizer_class': self.optimizer.__class__.__name__
            },
            
            # Scheduler state: learning rate scheduling
            'scheduler_state': {
                'state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'scheduler_class': self.scheduler.__class__.__name__ if self.scheduler is not None else None
            },
            
            # Mixed precision state: gradient scaling for AMP
            'scaler_state': {
                'state_dict': self.scaler.state_dict() if self.scaler is not None else None,
                'enabled': self.use_amp
            },
            
            # Training progress state: epoch, metrics, and global counters
            'training_state': {
                'epoch': self.current_epoch,  # Keep internal epoch as 0-based for consistency
                'best_val_metric': self.best_val_metric,
            },
            
            # Configuration state: model and training configuration
            'config_state': {
                'config': self.config
            },
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Modular checkpoint saved to {filepath}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['common']['ResidueEncoder_model_path'], "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model checkpoint saved to {best_path}")


    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        

        is_modular = ('model_state' in checkpoint and 'training_state' in checkpoint)
        # New format
        if is_modular:
            model_state_dict = checkpoint['model_state']['state_dict']
            
            if self.optimizer is not None and 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state']['state_dict'])
            
            if self.scheduler is not None and 'scheduler_state' in checkpoint:
                if checkpoint['scheduler_state']['state_dict'] is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state']['state_dict'])
            
            if self.scaler is not None and 'scaler_state' in checkpoint:
                if checkpoint['scaler_state']['state_dict'] is not None:
                    self.scaler.load_state_dict(checkpoint['scaler_state']['state_dict'])
            
            training_state = checkpoint['training_state']
            self.current_epoch = training_state['epoch'] + 1
            self.best_val_metric = training_state.get('best_val_metric', 0.0)
        # Legacy format
        else:
            model_state_dict = checkpoint['model_state_dict']
            
            if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_metric = checkpoint.get('best_metric', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
        
        # =========================================== Top@Handle DDP and torch.compile prefix mismatch ======================================================
        current_model_keys = list(self.model.state_dict().keys())
        checkpoint_keys = list(model_state_dict.keys())
        
        if len(current_model_keys) > 0 and len(checkpoint_keys) > 0:
            # Detect wrapper status for the current model
            current_has_ddp = current_model_keys[0].startswith('module.')
            current_has_compile = '_orig_mod.' in current_model_keys[0]

            # Detect wrapper status for the checkpoint
            checkpoint_has_ddp = checkpoint_keys[0].startswith('module.')
            checkpoint_has_compile = '_orig_mod.' in checkpoint_keys[0]

            # Logic to align prefixes. We first create a "clean" state_dict by stripping all known prefixes
            # from the checkpoint, and then add the prefixes required by the current model.
            
            # Step 1: Create a clean state_dict by stripping all prefixes from the checkpoint keys.
            clean_state_dict = {}
            for k, v in model_state_dict.items():
                temp_k = k
                if checkpoint_has_ddp:
                    temp_k = temp_k.replace('module.', '', 1)
                if checkpoint_has_compile:
                    temp_k = temp_k.replace('_orig_mod.', '', 1)
                clean_state_dict[temp_k] = v
            
            # Step 2: Add the necessary prefixes to the clean state_dict to match the current model.
            final_state_dict = {}
            for k, v in clean_state_dict.items():
                temp_k = k
                if current_has_compile:
                    temp_k = '_orig_mod.' + temp_k
                if current_has_ddp:
                    temp_k = 'module.' + temp_k
                final_state_dict[temp_k] = v
            
            model_state_dict = final_state_dict
        # =========================================== Bottom@Handle DDP and torch.compile prefix mismatch ===================================================

        # Load model state, now with correctly aligned keys
        try:
            self.model.load_state_dict(model_state_dict, strict=True)
            print(f"Successfully loaded model state_dict from {checkpoint_path}")
        except RuntimeError as e:
            # Enhanced error logging for easier debugging
            print(f"Error loading state_dict from {checkpoint_path}: {e}")
            print("This may be due to a model architecture mismatch or key-matching failure.")
            current_keys = set(self.model.state_dict().keys())
            loaded_keys = set(model_state_dict.keys())
            missing_keys = current_keys - loaded_keys
            unexpected_keys = loaded_keys - current_keys
            if missing_keys:
                print(f"Missing keys in loaded state_dict: {list(missing_keys)[:5]}...")
            if unexpected_keys:
                print(f"Unexpected keys in loaded state_dict: {list(unexpected_keys)[:5]}...")
            raise e
        
        # Memory cleanup - release checkpoint data
        del checkpoint, model_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resumed from epoch {self.current_epoch + 1}")



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

        if isinstance(batch, dgl.DGLGraph):
            return batch.to(self.device)
        
        # The most common case: a torch.Tensor.
        # Using isinstance is a slight performance optimization over hasattr for this frequent check.
        if isinstance(batch, torch.Tensor):
            try:
                # Use non_blocking for performance gains with pinned memory.
                return batch.to(self.device, non_blocking=True)
            except TypeError:
                # Fallback for older PyTorch versions or specific tensor types
                # that might not support non_blocking.
                return batch.to(self.device)

        # Generic handling for any object that implements a `.to()` method (duck-typing).
        # This covers DGL graphs, TorchDrug's PackedGraph, HuggingFace's BatchEncoding, etc.
        # We also ensure it's not a Module, which should not be moved this way.
        if hasattr(batch, "to") and callable(getattr(batch, "to")) and not isinstance(batch, nn.Module):
            try:
                # EAFP (Easier to Ask for Forgiveness than Permission) approach.
                return batch.to(self.device, non_blocking=True)
            except TypeError:
                # If the '.to' method doesn't support 'non_blocking', try without it.
                return batch.to(self.device)
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
            return torch.from_numpy(batch).to(self.device, non_blocking=True)
        
        # If the object type is not recognized, return it unmodified.
        return batch



class ResidueEncoderBetaTrainer:

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 传入ResidueEncoderBetaTrainer的config是总配置文件
        """
        self.config = config

        os.makedirs(self.config['common']['ResidueEncoder_model_path'], exist_ok=True)
        self.device = self.config['common']['device']
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32
        self.scaler = GradScaler() if self.use_amp else None
        self.best_val_metric = 0.0
        self.current_epoch = 0

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_loss_functions()
        self._setup_optimizations()



    def _setup_model(self):
        print("Initializing ResidueEncoderBeta model...")
        self.model = ResidueEncoderBeta(self.config['model']['ResidueEncoderBeta']).to(self.device)
        self.model.float()

        # =========================================== Top@unfreeze ESM2 layers ======================================================
        # 从配置中获取要解冻的层列表，如果未定义则默认为空列表 (保证向后兼容)
        unfreeze_layers_config = self.config['train']['ResidueEncoderBetaTrainer']['unfreeze_esm_layers']
        if unfreeze_layers_config:
            print(f"Unfreezing ESM layers based on config: {unfreeze_layers_config}")

            # 定位到真正的ESM模块 (nn.Module)
            # 路径: self.model (EvoSiteBeta) -> self.model.esm_model (EvolutionaryScaleModelingBeta) -> self.model.esm_model.model (ESM2)
            esm_module_to_unfreeze = self.model.esm_model.model
            
            # Case 1: 解冻所有层
            if unfreeze_layers_config == "all":
                for param in esm_module_to_unfreeze.parameters():
                    param.requires_grad = True
                if self.is_main_process:
                    print("All ESM model parameters have been unfrozen.")
            # Case 2: 解冻指定层
            elif isinstance(unfreeze_layers_config, list) and len(unfreeze_layers_config) > 0:
                unfrozen_params_count = 0
                
                # 直接通过模块访问来解冻指定的Transformer层
                for layer_idx in unfreeze_layers_config:
                    # 配置是1-based, ModuleList是0-based
                    if 0 <= layer_idx - 1 < len(esm_module_to_unfreeze.layers):
                        target_layer = esm_module_to_unfreeze.layers[layer_idx - 1]
                        for param in target_layer.parameters():
                            param.requires_grad = True
                            unfrozen_params_count += 1
                    else:
                        print(f"Layer index {layer_idx} is out of range for ESM model. Skipping.")
                
                # 解冻最终的LayerNorm层，这是一个好习惯
                for param in esm_module_to_unfreeze.emb_layer_norm_after.parameters():
                    param.requires_grad = True
                    unfrozen_params_count += param.numel() # param.numel()更准确地计数参数量
                
                trainable_params_in_esm = sum(p.numel() for p in esm_module_to_unfreeze.parameters() if p.requires_grad)
                print(f"Unfrozen {trainable_params_in_esm:,} parameters in specified ESM layers and final LayerNorm.")
        else:
            print("ESM model remains fully frozen (default behavior).")
        # =========================================== Bottom@unfreeze ESM2 layers =================================================== 

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model initialized with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")



    def _setup_data(self):
        print("Setting up datasets for ResidueEncoderBeta using EnzymeActiveSiteDataset3...")
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
            filter_incomplete_ec=False,
        )

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
            filter_incomplete_ec=False,
        )
        

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['train']['ResidueEncoderBetaTrainer']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['ResidueEncoderBetaTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=self.config['train']['ResidueEncoderBetaTrainer']['pin_memory'],
            prefetch_factor=4,
            drop_last=False
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config['train']['ResidueEncoderBetaTrainer']['batch_size'],
            shuffle=False,
            num_workers=self.config['train']['ResidueEncoderBetaTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=self.config['train']['ResidueEncoderBetaTrainer']['pin_memory'],
            prefetch_factor=4,
            drop_last=False
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.valid_dataset)}")
        print(f"Training batches per GPU: {len(self.train_loader)}")
        print(f"Validation batches per GPU: {len(self.valid_loader)}")



    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler with detailed parameter logging."""
        
        # =========================================== Top@PARAMETER ANALYSIS ======================================================
        print("Analyzing model parameters...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = total_params - trainable_count
        
        print(f"OVERALL PARAMETER STATISTICS:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_count:,}")
        print(f"  Frozen parameters: {frozen_count:,}")
        print(f"  Trainable ratio: {trainable_count/total_params:.2%}")
        
        trainable_params_by_component = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                component = name.split('.')[0]
                if component not in trainable_params_by_component:
                    trainable_params_by_component[component] = []
                trainable_params_by_component[component].append(name)
        for component, param_names in sorted(trainable_params_by_component.items()):
            print(f"  {component.upper()} - {len(param_names)} parameters:")
            for name in param_names:
                print(f"    - {name}")
        # =========================================== Bottom@PARAMETER ANALYSIS ===================================================
        

        # Setup optimizer
        optimizer_type = self.config['train']['ResidueEncoderBetaTrainer']['optimizer']
        lr = float(self.config['train']['ResidueEncoderBetaTrainer']['learning_rate'])
        weight_decay = float(self.config['train']['ResidueEncoderBetaTrainer']['weight_decay'])
        
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        

        # Setup scheduler (step-based for better control with gradient accumulation)
        scheduler_type = self.config['train']['ResidueEncoderBetaTrainer']['scheduler']
        if scheduler_type != 'none':
            # Calculate total training steps considering gradient accumulation
            steps_per_epoch = len(self.train_loader) // self.config['train']['ResidueEncoderBetaTrainer']['accumulation_steps']
            total_training_steps = steps_per_epoch * self.config['train']['ResidueEncoderBetaTrainer']['epochs']
            warmup_steps = int(steps_per_epoch * self.config['train']['ResidueEncoderBetaTrainer']['warmup_epochs'])
            self.scheduler = get_step_based_scheduler(
                self.optimizer,
                scheduler_type=scheduler_type,
                total_steps=total_training_steps,
                warmup_steps=warmup_steps,
                min_lr=self.config['train']['ResidueEncoderBetaTrainer']['min_lr']
            )
            
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Total training steps: {total_training_steps}")
            print(f"  Warmup steps: {warmup_steps}")
        else:
            self.scheduler = None
        
        # log
        print(f"\nOPTIMIZER SETUP:")
        print(f"  Optimizer: {optimizer_type}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Scheduler: {scheduler_type}")
        print(f"  Parameters in optimizer: {len(trainable_params)} tensors")
        print(f"  Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")



    def _setup_loss_functions(self):
        """Setup the primary classification loss function for EvoSiteBeta."""
        self.loss_fn = create_loss_function(
            loss_type=self.config['train']['ResidueEncoderBetaTrainer']['loss_type'],
            class_weights=self.config['train']['ResidueEncoderBetaTrainer']['class_weights'],
            label_smoothing=self.config['train']['ResidueEncoderBetaTrainer']['label_smoothing']
        ).to(self.device)

        print(f"Loss function initialized: {self.config['train']['ResidueEncoderBetaTrainer']['loss_type']}")



    def _setup_optimizations(self):
        print("Setting up training environment optimizations...")
        
        if torch.cuda.is_available():
            try:
                # cuDNN settings
                enable_cudnn_benchmark = self.config['train']['ResidueEncoderBetaTrainer']['cudnn_benchmark']
                if enable_cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True
                
                # Attention backend settings (for PyTorch 2.x+)
                use_efficient_backends = self.config['train']['ResidueEncoderBetaTrainer']['use_efficient_attention_backends']
                if use_efficient_backends:
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                else:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                
                # TF32 precision settings (for Ampere GPUs+)
                use_tf32 = self.config['train']['ResidueEncoderBetaTrainer']['use_tf32']
                if use_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    
            except Exception as e:
                print(f"部分训练环境优化设置失败: {e}")
        
        # Memory management settings
        self.enable_memory_optimization = self.config['train']['ResidueEncoderBetaTrainer']['enable_memory_optimization']
        self.memory_cleanup_frequency = self.config['train']['ResidueEncoderBetaTrainer']['memory_cleanup_frequency']



    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Training for {self.config['train']['ResidueEncoderBetaTrainer']['epochs']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['train']['ResidueEncoderBetaTrainer']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
        
            # Validation 
            if (epoch + 1) % self.config['train']['ResidueEncoderBetaTrainer']['val_frequency'] == 0:
                val_metrics = self.validate_epoch()
                print(f"[Epoch {epoch + 1}] Train - avg_loss: {train_losses['avg_train_loss']:.4f}, total_loss: {train_losses['total_train_loss']:.4f}")
                print(f"[Epoch {epoch + 1}] Val Metrics - {', '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])}")
                
                # Check for best model - use correct metric name based on num_classes
                metric_name = 'mcc' if self.config['model']['ResidueEncoderBeta']['num_class'] == 2 else 'multi_class_mcc'
                current_metric = val_metrics[metric_name]
                is_best = current_metric > self.best_val_metric
                
                if is_best:
                    self.best_val_metric = current_metric
                    print(f"New best model! MCC: {current_metric:.4f}")

                # Save checkpoint
                if (epoch + 1) % self.config['train']['ResidueEncoderBetaTrainer']['save_frequency'] == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
            else:
                print(f"[Epoch {epoch + 1}] Train - avg_loss: {train_losses['avg_train_loss']:.4f}, total_loss: {train_losses['total_train_loss']:.4f}")
            
        
        # Training completed
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation metric: {self.best_val_metric:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(filename="final_checkpoint.pt")




    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        accumulation_steps = self.config['train']['ResidueEncoderBetaTrainer']['accumulation_steps']
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config['train']['ResidueEncoderBetaTrainer']['epochs']} [TRAIN]", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            batch_failed = False
            try:
                if batch is None:   raise ValueError("Batch is None, likely due to a collate_fn error.")
                batch = self._move_batch_to_device(batch)
                with autocast(enabled=self.use_amp, dtype=self.dtype):
                    logits, protein_mask = self.model(batch)
                    labels = batch['targets'].long()
                    assert logits.size(0) == labels.size(0), f"Mismatch between number of residues in logits ({logits.size(0)}) and targets ({labels.size(0)}). "
                    main_loss = self.loss_fn(logits, labels)
                    loss = main_loss / accumulation_steps
                    if not self._check_loss_validity(loss, f"batch_{batch_idx}"):   raise ValueError(f"Invalid loss detected: {loss}")
                    
            except Exception as e:
                batch_failed = True
                uniprot_ids = batch.get('uniprot_ids', ['N/A'] * self.config['train']['ResidueEncoderBetaTrainer']['batch_size'])
                print(f"Training batch {batch_idx} failed and will be skipped. Error: {e}")
                print(f"Problematic UniProt IDs might be in: {uniprot_ids}")
                sys.stdout.flush()
                sys.stderr.flush()

            if batch_failed:
                continue
            
            try:
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError as e:
                if "SoftmaxBackward" in str(e):
                    print("NaN in SoftmaxBackward, skip batch.")
                    continue
                if "returned nan values in its" in str(e):
                    print(f"[Warning] NaN detected in backward at batch {batch_idx}: {str(e)}")
                    self.optimizer.zero_grad()
                    if self.use_amp and self.scaler is not None:
                        # 给 GradScaler 一个机会调整 scale，减小后续溢出的概率
                        self.scaler.update()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    continue
                raise
            total_loss += main_loss.item()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['ResidueEncoderBetaTrainer']['gradient_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['ResidueEncoderBetaTrainer']['gradient_clip'])
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()


            pbar.set_postfix(loss=f"{main_loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Return the average loss for the epoch.
        avg_loss = total_loss / num_batches
        return {'avg_train_loss': avg_loss, 'total_train_loss': total_loss}



    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch，返回分类指标。"""
        self.model.eval()
        all_metrics = defaultdict(list)
        num_classes = self.config['model']['ResidueEncoderBeta']['num_class']
        
        pbar = tqdm(
            enumerate(self.valid_loader),
            total=len(self.valid_loader),
            desc=f"Epoch {self.current_epoch + 1}/{self.config['train']['ResidueEncoderBetaTrainer']['epochs']} [VAL]",
            unit="batch"
        )
        
        try:
            for batch_idx, batch in pbar:
                if batch_idx >= len(self.valid_loader):
                    break
                if batch is None:
                    continue

                try:
                    batch = self._move_batch_to_device(batch)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        flat_logits, protein_mask = self.model(batch)
                        flat_labels = batch['targets'].long()
                        assert flat_logits.size(0) == flat_labels.size(0), f"Mismatch: logits({flat_logits.size(0)}) vs targets({flat_labels.size(0)})"
                except Exception as e:
                    uniprot_ids = batch.get('uniprot_ids', ['N/A'])
                    print(f"Validation batch {batch_idx} failed. Error: {e}, UniProt IDs: {uniprot_ids}")
                    continue

                # Data Format Adaptation: 将flat格式转换为batch格式
                batch_size, max_len = protein_mask.shape
                labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
                logits = torch.zeros(batch_size, max_len, num_classes, device=self.device)
                labels[protein_mask.bool()] = flat_labels
                logits[protein_mask.bool()] = flat_logits
                valid_mask = protein_mask

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

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
        finally:
            if hasattr(pbar, 'close'):
                pbar.close()

        # 计算所有指标的平均值
        classification_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                classification_metrics[metric_name] = sum(values) / len(values)
        for metric_name, value in sorted(classification_metrics.items()):
            print(f"  {metric_name}: {value:.4f}")
        return classification_metrics



    def _check_loss_validity(self, loss, step_name: str = "") -> bool:
        """
        简单检查 loss 是否为有效数值。
        返回:
            True  : loss 正常
            False : loss 为 NaN 或 Inf
        """
        # 注意：loss 这里应该是标量 tensor
        if torch.isnan(loss) or torch.isinf(loss):
            msg = f"{step_name}: Invalid loss detected {loss}"
            print(msg)
            return False

        return True
    


    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model checkpoint
            filename: Optional checkpoint filename (defaults to epoch-based naming)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pt"      # Use 1-based epoch numbering for checkpoint filename
        filepath = os.path.join(self.config['common']['ResidueEncoder_model_path'], filename)

        model_to_save = self.model

        
        checkpoint = {
            # Model state: core model parameters
            'model_state': {
                'state_dict': model_to_save.state_dict(),
                'model_class': model_to_save.__class__.__name__
            },
            
            # Optimizer state: optimization parameters and momentum
            'optimizer_state': {
                'state_dict': self.optimizer.state_dict(),
                'optimizer_class': self.optimizer.__class__.__name__
            },
            
            # Scheduler state: learning rate scheduling
            'scheduler_state': {
                'state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'scheduler_class': self.scheduler.__class__.__name__ if self.scheduler is not None else None
            },
            
            # Mixed precision state: gradient scaling for AMP
            'scaler_state': {
                'state_dict': self.scaler.state_dict() if self.scaler is not None else None,
                'enabled': self.use_amp
            },
            
            # Training progress state: epoch, metrics, and global counters
            'training_state': {
                'epoch': self.current_epoch,  # Keep internal epoch as 0-based for consistency
                'best_val_metric': self.best_val_metric,
            },
            
            # Configuration state: model and training configuration
            'config_state': {
                'config': self.config
            },
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Modular checkpoint saved to {filepath}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['common']['ResidueEncoder_model_path'], "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model checkpoint saved to {best_path}")


    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        

        is_modular = ('model_state' in checkpoint and 'training_state' in checkpoint)
        # New format
        if is_modular:
            model_state_dict = checkpoint['model_state']['state_dict']
            
            if self.optimizer is not None and 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state']['state_dict'])
            
            if self.scheduler is not None and 'scheduler_state' in checkpoint:
                if checkpoint['scheduler_state']['state_dict'] is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state']['state_dict'])
            
            if self.scaler is not None and 'scaler_state' in checkpoint:
                if checkpoint['scaler_state']['state_dict'] is not None:
                    self.scaler.load_state_dict(checkpoint['scaler_state']['state_dict'])
            
            training_state = checkpoint['training_state']
            self.current_epoch = training_state['epoch'] + 1
            self.best_val_metric = training_state.get('best_val_metric', 0.0)
        # Legacy format
        else:
            model_state_dict = checkpoint['model_state_dict']
            
            if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_metric = checkpoint.get('best_metric', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
        
        # =========================================== Top@Handle DDP and torch.compile prefix mismatch ======================================================
        current_model_keys = list(self.model.state_dict().keys())
        checkpoint_keys = list(model_state_dict.keys())
        
        if len(current_model_keys) > 0 and len(checkpoint_keys) > 0:
            # Detect wrapper status for the current model
            current_has_ddp = current_model_keys[0].startswith('module.')
            current_has_compile = '_orig_mod.' in current_model_keys[0]

            # Detect wrapper status for the checkpoint
            checkpoint_has_ddp = checkpoint_keys[0].startswith('module.')
            checkpoint_has_compile = '_orig_mod.' in checkpoint_keys[0]

            # Logic to align prefixes. We first create a "clean" state_dict by stripping all known prefixes
            # from the checkpoint, and then add the prefixes required by the current model.
            
            # Step 1: Create a clean state_dict by stripping all prefixes from the checkpoint keys.
            clean_state_dict = {}
            for k, v in model_state_dict.items():
                temp_k = k
                if checkpoint_has_ddp:
                    temp_k = temp_k.replace('module.', '', 1)
                if checkpoint_has_compile:
                    temp_k = temp_k.replace('_orig_mod.', '', 1)
                clean_state_dict[temp_k] = v
            
            # Step 2: Add the necessary prefixes to the clean state_dict to match the current model.
            final_state_dict = {}
            for k, v in clean_state_dict.items():
                temp_k = k
                if current_has_compile:
                    temp_k = '_orig_mod.' + temp_k
                if current_has_ddp:
                    temp_k = 'module.' + temp_k
                final_state_dict[temp_k] = v
            
            model_state_dict = final_state_dict
        # =========================================== Bottom@Handle DDP and torch.compile prefix mismatch ===================================================

        # Load model state, now with correctly aligned keys
        try:
            self.model.load_state_dict(model_state_dict, strict=True)
            print(f"Successfully loaded model state_dict from {checkpoint_path}")
        except RuntimeError as e:
            # Enhanced error logging for easier debugging
            print(f"Error loading state_dict from {checkpoint_path}: {e}")
            print("This may be due to a model architecture mismatch or key-matching failure.")
            current_keys = set(self.model.state_dict().keys())
            loaded_keys = set(model_state_dict.keys())
            missing_keys = current_keys - loaded_keys
            unexpected_keys = loaded_keys - current_keys
            if missing_keys:
                print(f"Missing keys in loaded state_dict: {list(missing_keys)[:5]}...")
            if unexpected_keys:
                print(f"Unexpected keys in loaded state_dict: {list(unexpected_keys)[:5]}...")
            raise e
        
        # Memory cleanup - release checkpoint data
        del checkpoint, model_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resumed from epoch {self.current_epoch + 1}")



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

        if isinstance(batch, dgl.DGLGraph):
            return batch.to(self.device)
        
        # The most common case: a torch.Tensor.
        # Using isinstance is a slight performance optimization over hasattr for this frequent check.
        if isinstance(batch, torch.Tensor):
            try:
                # Use non_blocking for performance gains with pinned memory.
                return batch.to(self.device, non_blocking=True)
            except TypeError:
                # Fallback for older PyTorch versions or specific tensor types
                # that might not support non_blocking.
                return batch.to(self.device)

        # Generic handling for any object that implements a `.to()` method (duck-typing).
        # This covers DGL graphs, TorchDrug's PackedGraph, HuggingFace's BatchEncoding, etc.
        # We also ensure it's not a Module, which should not be moved this way.
        if hasattr(batch, "to") and callable(getattr(batch, "to")) and not isinstance(batch, nn.Module):
            try:
                # EAFP (Easier to Ask for Forgiveness than Permission) approach.
                return batch.to(self.device, non_blocking=True)
            except TypeError:
                # If the '.to' method doesn't support 'non_blocking', try without it.
                return batch.to(self.device)
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
            return torch.from_numpy(batch).to(self.device, non_blocking=True)
        
        # If the object type is not recognized, return it unmodified.
        return batch



class ResidueEncoderGammaTrainer:

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 传入ResidueEncoderGammaTrainer的config是总配置文件
        """
        self.config = config

        os.makedirs(self.config['common']['ResidueEncoder_model_path'], exist_ok=True)
        self.device = self.config['common']['device']
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32
        self.scaler = GradScaler() if self.use_amp else None
        self.best_val_metric = 0.0
        self.current_epoch = 0

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_loss_functions()
        self._setup_optimizations()



    def _setup_model(self):
        print("Initializing ResidueEncoderGamma model...")
        self.model = ResidueEncoderGamma(self.config['model']['ResidueEncoderGamma']).to(self.device)
        self.model.float()

        # =========================================== Top@unfreeze ESM2 layers ======================================================
        # 从配置中获取要解冻的层列表，如果未定义则默认为空列表 (保证向后兼容)
        unfreeze_layers_config = self.config['train']['ResidueEncoderGammaTrainer']['unfreeze_esm_layers']
        if unfreeze_layers_config:
            print(f"Unfreezing ESM layers based on config: {unfreeze_layers_config}")

            # 定位到真正的ESM模块 (nn.Module)
            # 路径: self.model (EvoSiteBeta) -> self.model.esm_model (EvolutionaryScaleModelingBeta) -> self.model.esm_model.model (ESM2)
            esm_module_to_unfreeze = self.model.esm_model.model
            
            # Case 1: 解冻所有层
            if unfreeze_layers_config == "all":
                for param in esm_module_to_unfreeze.parameters():
                    param.requires_grad = True
                if self.is_main_process:
                    print("All ESM model parameters have been unfrozen.")
            # Case 2: 解冻指定层
            elif isinstance(unfreeze_layers_config, list) and len(unfreeze_layers_config) > 0:
                unfrozen_params_count = 0
                
                # 直接通过模块访问来解冻指定的Transformer层
                for layer_idx in unfreeze_layers_config:
                    # 配置是1-based, ModuleList是0-based
                    if 0 <= layer_idx - 1 < len(esm_module_to_unfreeze.layers):
                        target_layer = esm_module_to_unfreeze.layers[layer_idx - 1]
                        for param in target_layer.parameters():
                            param.requires_grad = True
                            unfrozen_params_count += 1
                    else:
                        print(f"Layer index {layer_idx} is out of range for ESM model. Skipping.")
                
                # 解冻最终的LayerNorm层，这是一个好习惯
                for param in esm_module_to_unfreeze.emb_layer_norm_after.parameters():
                    param.requires_grad = True
                    unfrozen_params_count += param.numel() # param.numel()更准确地计数参数量
                
                trainable_params_in_esm = sum(p.numel() for p in esm_module_to_unfreeze.parameters() if p.requires_grad)
                print(f"Unfrozen {trainable_params_in_esm:,} parameters in specified ESM layers and final LayerNorm.")
        else:
            print("ESM model remains fully frozen (default behavior).")
        # =========================================== Bottom@unfreeze ESM2 layers =================================================== 

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model initialized with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")



    def _setup_data(self):
        print("Setting up datasets for ResidueEncoderGamma using EnzymeActiveSiteDataset3...")
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
            filter_incomplete_ec=False,
        )

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
            filter_incomplete_ec=False,
        )
        

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['train']['ResidueEncoderGammaTrainer']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['ResidueEncoderGammaTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=self.config['train']['ResidueEncoderGammaTrainer']['pin_memory'],
            prefetch_factor=4,
            drop_last=False
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config['train']['ResidueEncoderGammaTrainer']['batch_size'],
            shuffle=False,
            num_workers=self.config['train']['ResidueEncoderGammaTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=self.config['train']['ResidueEncoderGammaTrainer']['pin_memory'],
            prefetch_factor=4,
            drop_last=False
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.valid_dataset)}")
        print(f"Training batches per GPU: {len(self.train_loader)}")
        print(f"Validation batches per GPU: {len(self.valid_loader)}")



    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler with detailed parameter logging."""
        
        # =========================================== Top@PARAMETER ANALYSIS ======================================================
        print("Analyzing model parameters...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = total_params - trainable_count
        
        print(f"OVERALL PARAMETER STATISTICS:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_count:,}")
        print(f"  Frozen parameters: {frozen_count:,}")
        print(f"  Trainable ratio: {trainable_count/total_params:.2%}")
        
        trainable_params_by_component = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                component = name.split('.')[0]
                if component not in trainable_params_by_component:
                    trainable_params_by_component[component] = []
                trainable_params_by_component[component].append(name)
        for component, param_names in sorted(trainable_params_by_component.items()):
            print(f"  {component.upper()} - {len(param_names)} parameters:")
            for name in param_names:
                print(f"    - {name}")
        # =========================================== Bottom@PARAMETER ANALYSIS ===================================================
        

        # Setup optimizer
        optimizer_type = self.config['train']['ResidueEncoderGammaTrainer']['optimizer']
        lr = float(self.config['train']['ResidueEncoderGammaTrainer']['learning_rate'])
        weight_decay = float(self.config['train']['ResidueEncoderGammaTrainer']['weight_decay'])
        
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        

        # Setup scheduler (step-based for better control with gradient accumulation)
        scheduler_type = self.config['train']['ResidueEncoderGammaTrainer']['scheduler']
        if scheduler_type != 'none':
            # Calculate total training steps considering gradient accumulation
            steps_per_epoch = len(self.train_loader) // self.config['train']['ResidueEncoderGammaTrainer']['accumulation_steps']
            total_training_steps = steps_per_epoch * self.config['train']['ResidueEncoderGammaTrainer']['epochs']
            warmup_steps = int(steps_per_epoch * self.config['train']['ResidueEncoderGammaTrainer']['warmup_epochs'])
            self.scheduler = get_step_based_scheduler(
                self.optimizer,
                scheduler_type=scheduler_type,
                total_steps=total_training_steps,
                warmup_steps=warmup_steps,
                min_lr=self.config['train']['ResidueEncoderGammaTrainer']['min_lr']
            )
            
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Total training steps: {total_training_steps}")
            print(f"  Warmup steps: {warmup_steps}")
        else:
            self.scheduler = None
        
        # log
        print(f"\nOPTIMIZER SETUP:")
        print(f"  Optimizer: {optimizer_type}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Scheduler: {scheduler_type}")
        print(f"  Parameters in optimizer: {len(trainable_params)} tensors")
        print(f"  Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")



    def _setup_loss_functions(self):
        """Setup the primary classification loss function for EvoSiteBeta."""
        self.loss_fn = create_loss_function(
            loss_type=self.config['train']['ResidueEncoderGammaTrainer']['loss_type'],
            class_weights=self.config['train']['ResidueEncoderGammaTrainer']['class_weights'],
            label_smoothing=self.config['train']['ResidueEncoderGammaTrainer']['label_smoothing']
        ).to(self.device)

        print(f"Loss function initialized: {self.config['train']['ResidueEncoderGammaTrainer']['loss_type']}")



    def _setup_optimizations(self):
        print("Setting up training environment optimizations...")
        
        if torch.cuda.is_available():
            try:
                # cuDNN settings
                enable_cudnn_benchmark = self.config['train']['ResidueEncoderGammaTrainer']['cudnn_benchmark']
                if enable_cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True
                
                # Attention backend settings (for PyTorch 2.x+)
                use_efficient_backends = self.config['train']['ResidueEncoderGammaTrainer']['use_efficient_attention_backends']
                if use_efficient_backends:
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                else:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                
                # TF32 precision settings (for Ampere GPUs+)
                use_tf32 = self.config['train']['ResidueEncoderGammaTrainer']['use_tf32']
                if use_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    
            except Exception as e:
                print(f"部分训练环境优化设置失败: {e}")
        
        # Memory management settings
        self.enable_memory_optimization = self.config['train']['ResidueEncoderGammaTrainer']['enable_memory_optimization']
        self.memory_cleanup_frequency = self.config['train']['ResidueEncoderGammaTrainer']['memory_cleanup_frequency']



    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Training for {self.config['train']['ResidueEncoderGammaTrainer']['epochs']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['train']['ResidueEncoderGammaTrainer']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
        
            # Validation 
            if (epoch + 1) % self.config['train']['ResidueEncoderGammaTrainer']['val_frequency'] == 0:
                val_metrics = self.validate_epoch()
                print(f"[Epoch {epoch + 1}] Train - avg_loss: {train_losses['avg_train_loss']:.4f}, total_loss: {train_losses['total_train_loss']:.4f}")
                print(f"[Epoch {epoch + 1}] Val Metrics - {', '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])}")
                
                # Check for best model - use correct metric name based on num_classes
                metric_name = 'mcc' if self.config['model']['ResidueEncoderGamma']['num_class'] == 2 else 'multi_class_mcc'
                current_metric = val_metrics[metric_name]
                is_best = current_metric > self.best_val_metric
                
                if is_best:
                    self.best_val_metric = current_metric
                    print(f"New best model! MCC: {current_metric:.4f}")

                # Save checkpoint
                if (epoch + 1) % self.config['train']['ResidueEncoderGammaTrainer']['save_frequency'] == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
            else:
                print(f"[Epoch {epoch + 1}] Train - avg_loss: {train_losses['avg_train_loss']:.4f}, total_loss: {train_losses['total_train_loss']:.4f}")
            
        
        # Training completed
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation metric: {self.best_val_metric:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(filename="final_checkpoint.pt")




    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        accumulation_steps = self.config['train']['ResidueEncoderGammaTrainer']['accumulation_steps']
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config['train']['ResidueEncoderGammaTrainer']['epochs']} [TRAIN]", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            batch_failed = False
            try:
                if batch is None:   raise ValueError("Batch is None, likely due to a collate_fn error.")
                batch = self._move_batch_to_device(batch)
                with autocast(enabled=self.use_amp, dtype=self.dtype):
                    logits, protein_mask = self.model(batch)
                    labels = batch['targets'].long()
                    assert logits.size(0) == labels.size(0), f"Mismatch between number of residues in logits ({logits.size(0)}) and targets ({labels.size(0)}). "
                    main_loss = self.loss_fn(logits, labels)
                    loss = main_loss / accumulation_steps
                    if not self._check_loss_validity(loss, f"batch_{batch_idx}"):   raise ValueError(f"Invalid loss detected: {loss}")
                    
            except Exception as e:
                batch_failed = True
                uniprot_ids = batch.get('uniprot_ids', ['N/A'] * self.config['train']['ResidueEncoderGammaTrainer']['batch_size'])
                print(f"Training batch {batch_idx} failed and will be skipped. Error: {e}")
                print(f"Problematic UniProt IDs might be in: {uniprot_ids}")
                sys.stdout.flush()
                sys.stderr.flush()

            if batch_failed:
                continue
            
            try:
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError as e:
                if "SoftmaxBackward" in str(e):
                    print("NaN in SoftmaxBackward, skip batch.")
                    continue
                if "returned nan values in its" in str(e):
                    print(f"[Warning] NaN detected in backward at batch {batch_idx}: {str(e)}")
                    self.optimizer.zero_grad()
                    if self.use_amp and self.scaler is not None:
                        # 给 GradScaler 一个机会调整 scale，减小后续溢出的概率
                        self.scaler.update()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    continue
                raise
            total_loss += main_loss.item()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['ResidueEncoderGammaTrainer']['gradient_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['ResidueEncoderGammaTrainer']['gradient_clip'])
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()


            pbar.set_postfix(loss=f"{main_loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Return the average loss for the epoch.
        avg_loss = total_loss / num_batches
        return {'avg_train_loss': avg_loss, 'total_train_loss': total_loss}



    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch，返回分类指标。"""
        self.model.eval()
        all_metrics = defaultdict(list)
        num_classes = self.config['model']['ResidueEncoderGamma']['num_class']
        
        pbar = tqdm(
            enumerate(self.valid_loader),
            total=len(self.valid_loader),
            desc=f"Epoch {self.current_epoch + 1}/{self.config['train']['ResidueEncoderGammaTrainer']['epochs']} [VAL]",
            unit="batch"
        )
        
        try:
            for batch_idx, batch in pbar:
                if batch_idx >= len(self.valid_loader):
                    break
                if batch is None:
                    continue

                try:
                    batch = self._move_batch_to_device(batch)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        flat_logits, protein_mask = self.model(batch)
                        flat_labels = batch['targets'].long()
                        assert flat_logits.size(0) == flat_labels.size(0), f"Mismatch: logits({flat_logits.size(0)}) vs targets({flat_labels.size(0)})"
                except Exception as e:
                    uniprot_ids = batch.get('uniprot_ids', ['N/A'])
                    print(f"Validation batch {batch_idx} failed. Error: {e}, UniProt IDs: {uniprot_ids}")
                    continue

                # Data Format Adaptation: 将flat格式转换为batch格式
                batch_size, max_len = protein_mask.shape
                labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
                logits = torch.zeros(batch_size, max_len, num_classes, device=self.device)
                labels[protein_mask.bool()] = flat_labels
                logits[protein_mask.bool()] = flat_logits
                valid_mask = protein_mask

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

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
        finally:
            if hasattr(pbar, 'close'):
                pbar.close()

        # 计算所有指标的平均值
        classification_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                classification_metrics[metric_name] = sum(values) / len(values)
        for metric_name, value in sorted(classification_metrics.items()):
            print(f"  {metric_name}: {value:.4f}")
        return classification_metrics



    def _check_loss_validity(self, loss, step_name: str = "") -> bool:
        """
        简单检查 loss 是否为有效数值。
        返回:
            True  : loss 正常
            False : loss 为 NaN 或 Inf
        """
        # 注意：loss 这里应该是标量 tensor
        if torch.isnan(loss) or torch.isinf(loss):
            msg = f"{step_name}: Invalid loss detected {loss}"
            print(msg)
            return False

        return True
    


    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model checkpoint
            filename: Optional checkpoint filename (defaults to epoch-based naming)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pt"      # Use 1-based epoch numbering for checkpoint filename
        filepath = os.path.join(self.config['common']['ResidueEncoder_model_path'], filename)

        model_to_save = self.model

        
        checkpoint = {
            # Model state: core model parameters
            'model_state': {
                'state_dict': model_to_save.state_dict(),
                'model_class': model_to_save.__class__.__name__
            },
            
            # Optimizer state: optimization parameters and momentum
            'optimizer_state': {
                'state_dict': self.optimizer.state_dict(),
                'optimizer_class': self.optimizer.__class__.__name__
            },
            
            # Scheduler state: learning rate scheduling
            'scheduler_state': {
                'state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'scheduler_class': self.scheduler.__class__.__name__ if self.scheduler is not None else None
            },
            
            # Mixed precision state: gradient scaling for AMP
            'scaler_state': {
                'state_dict': self.scaler.state_dict() if self.scaler is not None else None,
                'enabled': self.use_amp
            },
            
            # Training progress state: epoch, metrics, and global counters
            'training_state': {
                'epoch': self.current_epoch,  # Keep internal epoch as 0-based for consistency
                'best_val_metric': self.best_val_metric,
            },
            
            # Configuration state: model and training configuration
            'config_state': {
                'config': self.config
            },
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Modular checkpoint saved to {filepath}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['common']['ResidueEncoder_model_path'], "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model checkpoint saved to {best_path}")


    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        

        is_modular = ('model_state' in checkpoint and 'training_state' in checkpoint)
        # New format
        if is_modular:
            model_state_dict = checkpoint['model_state']['state_dict']
            
            if self.optimizer is not None and 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state']['state_dict'])
            
            if self.scheduler is not None and 'scheduler_state' in checkpoint:
                if checkpoint['scheduler_state']['state_dict'] is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state']['state_dict'])
            
            if self.scaler is not None and 'scaler_state' in checkpoint:
                if checkpoint['scaler_state']['state_dict'] is not None:
                    self.scaler.load_state_dict(checkpoint['scaler_state']['state_dict'])
            
            training_state = checkpoint['training_state']
            self.current_epoch = training_state['epoch'] + 1
            self.best_val_metric = training_state.get('best_val_metric', 0.0)
        # Legacy format
        else:
            model_state_dict = checkpoint['model_state_dict']
            
            if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_metric = checkpoint.get('best_metric', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
        
        # =========================================== Top@Handle DDP and torch.compile prefix mismatch ======================================================
        current_model_keys = list(self.model.state_dict().keys())
        checkpoint_keys = list(model_state_dict.keys())
        
        if len(current_model_keys) > 0 and len(checkpoint_keys) > 0:
            # Detect wrapper status for the current model
            current_has_ddp = current_model_keys[0].startswith('module.')
            current_has_compile = '_orig_mod.' in current_model_keys[0]

            # Detect wrapper status for the checkpoint
            checkpoint_has_ddp = checkpoint_keys[0].startswith('module.')
            checkpoint_has_compile = '_orig_mod.' in checkpoint_keys[0]

            # Logic to align prefixes. We first create a "clean" state_dict by stripping all known prefixes
            # from the checkpoint, and then add the prefixes required by the current model.
            
            # Step 1: Create a clean state_dict by stripping all prefixes from the checkpoint keys.
            clean_state_dict = {}
            for k, v in model_state_dict.items():
                temp_k = k
                if checkpoint_has_ddp:
                    temp_k = temp_k.replace('module.', '', 1)
                if checkpoint_has_compile:
                    temp_k = temp_k.replace('_orig_mod.', '', 1)
                clean_state_dict[temp_k] = v
            
            # Step 2: Add the necessary prefixes to the clean state_dict to match the current model.
            final_state_dict = {}
            for k, v in clean_state_dict.items():
                temp_k = k
                if current_has_compile:
                    temp_k = '_orig_mod.' + temp_k
                if current_has_ddp:
                    temp_k = 'module.' + temp_k
                final_state_dict[temp_k] = v
            
            model_state_dict = final_state_dict
        # =========================================== Bottom@Handle DDP and torch.compile prefix mismatch ===================================================

        # Load model state, now with correctly aligned keys
        try:
            self.model.load_state_dict(model_state_dict, strict=True)
            print(f"Successfully loaded model state_dict from {checkpoint_path}")
        except RuntimeError as e:
            # Enhanced error logging for easier debugging
            print(f"Error loading state_dict from {checkpoint_path}: {e}")
            print("This may be due to a model architecture mismatch or key-matching failure.")
            current_keys = set(self.model.state_dict().keys())
            loaded_keys = set(model_state_dict.keys())
            missing_keys = current_keys - loaded_keys
            unexpected_keys = loaded_keys - current_keys
            if missing_keys:
                print(f"Missing keys in loaded state_dict: {list(missing_keys)[:5]}...")
            if unexpected_keys:
                print(f"Unexpected keys in loaded state_dict: {list(unexpected_keys)[:5]}...")
            raise e
        
        # Memory cleanup - release checkpoint data
        del checkpoint, model_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resumed from epoch {self.current_epoch + 1}")



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

        if isinstance(batch, dgl.DGLGraph):
            return batch.to(self.device)
        
        # The most common case: a torch.Tensor.
        # Using isinstance is a slight performance optimization over hasattr for this frequent check.
        if isinstance(batch, torch.Tensor):
            try:
                # Use non_blocking for performance gains with pinned memory.
                return batch.to(self.device, non_blocking=True)
            except TypeError:
                # Fallback for older PyTorch versions or specific tensor types
                # that might not support non_blocking.
                return batch.to(self.device)

        # Generic handling for any object that implements a `.to()` method (duck-typing).
        # This covers DGL graphs, TorchDrug's PackedGraph, HuggingFace's BatchEncoding, etc.
        # We also ensure it's not a Module, which should not be moved this way.
        if hasattr(batch, "to") and callable(getattr(batch, "to")) and not isinstance(batch, nn.Module):
            try:
                # EAFP (Easier to Ask for Forgiveness than Permission) approach.
                return batch.to(self.device, non_blocking=True)
            except TypeError:
                # If the '.to' method doesn't support 'non_blocking', try without it.
                return batch.to(self.device)
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
            return torch.from_numpy(batch).to(self.device, non_blocking=True)
        
        # If the object type is not recognized, return it unmodified.
        return batch



class ResidueEncoderGammaBinaryTrainer(ResidueEncoderGammaTrainer):
    """
    Binary classification version of ResidueEncoderGammaTrainer.
    """
    def _setup_model(self):
        """Override to initialize the Binary model."""
        print("Initializing ResidueEncoderGammaBinary model...")
        self.model = ResidueEncoderGammaBinary(self.config['model']['ResidueEncoderGamma']).to(self.device)
        self.model.float()

        # Re-implement unfreezing logic (copied from parent since we override the whole method)
        unfreeze_layers_config = self.config['train']['ResidueEncoderGammaTrainer']['unfreeze_esm_layers']
        if unfreeze_layers_config:
            print(f"Unfreezing ESM layers based on config: {unfreeze_layers_config}")
            esm_module_to_unfreeze = self.model.esm_model.model
            if unfreeze_layers_config == "all":
                for param in esm_module_to_unfreeze.parameters():
                    param.requires_grad = True
                if self.is_main_process if hasattr(self, 'is_main_process') else True:
                    print("All ESM model parameters have been unfrozen.")
            elif isinstance(unfreeze_layers_config, list) and len(unfreeze_layers_config) > 0:
                unfrozen_params_count = 0
                for layer_idx in unfreeze_layers_config:
                    if 0 <= layer_idx - 1 < len(esm_module_to_unfreeze.layers):
                        target_layer = esm_module_to_unfreeze.layers[layer_idx - 1]
                        for param in target_layer.parameters():
                            param.requires_grad = True
                            unfrozen_params_count += 1
                for param in esm_module_to_unfreeze.emb_layer_norm_after.parameters():
                    param.requires_grad = True
                    unfrozen_params_count += param.numel()
                trainable_params_in_esm = sum(p.numel() for p in esm_module_to_unfreeze.parameters() if p.requires_grad)
                print(f"Unfrozen {trainable_params_in_esm:,} parameters in specified ESM layers and final LayerNorm.")
        else:
            print("ESM model remains fully frozen (default behavior).")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")

    def _setup_data(self):
        print("Setting up BINARY datasets for ResidueEncoderGamma...")
        self.train_dataset = EnzymeActiveSiteBinaryDataset3(
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
            filter_incomplete_ec=False,
        )

        self.valid_dataset = EnzymeActiveSiteBinaryDataset3(
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
            filter_incomplete_ec=False,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['train']['ResidueEncoderGammaTrainer']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['ResidueEncoderGammaTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=self.config['train']['ResidueEncoderGammaTrainer']['pin_memory'],
            prefetch_factor=4,
            drop_last=False
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config['train']['ResidueEncoderGammaTrainer']['batch_size'],
            shuffle=False,
            num_workers=self.config['train']['ResidueEncoderGammaTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=self.config['train']['ResidueEncoderGammaTrainer']['pin_memory'],
            prefetch_factor=4,
            drop_last=False
        )

        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.valid_dataset)}")
        print(f"Training batches per GPU: {len(self.train_loader)}")
        print(f"Validation batches per GPU: {len(self.valid_loader)}")



class SiteSupConTrainer:
    def __init__(self, config: Dict):

        self.config = config

        os.makedirs(self.config['common']['SiteSupCon_model_path'], exist_ok=True)
        self.device = torch.device(self.config['common']['device'])
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32
        self.scaler = GradScaler() if self.use_amp else None
        self.best_val_metric = -1.0
        self.current_epoch = 0

        # ========================================model========================================
        self.model = ResidueEncoderGamma(self.config['model']['ResidueEncoderGamma']).to(self.device)
        self.model.float()

        pretrained_dir_path = self.config['common']['ResidueEncoder_model_path']
        if pretrained_dir_path and os.path.isdir(pretrained_dir_path):
            try:
                pretrained_model_path = os.path.join(pretrained_dir_path, "best_model.pt")
                checkpoint = torch.load(pretrained_model_path, map_location=self.device)
                if 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']['state_dict']
                else:
                    state_dict = checkpoint
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if len(missing_keys) > 0:       print(f"  [Warning] Missing keys in backbone: {missing_keys[:5]} ... (Total: {len(missing_keys)})")
                if len(unexpected_keys) > 0:    print(f"  [Warning] Unexpected keys in checkpoint: {unexpected_keys[:5]} ... (Total: {len(unexpected_keys)})")
            except Exception as e:
                print(f"[SiteSupConTrainer] Error loading pretrained checkpoint: {e}")
        else:
            print("[SiteSupConTrainer] No 'pretrained_checkpoint_path' found in config or file does not exist.")
            print("[SiteSupConTrainer] Model will start with RANDOM initialization.")

        unfreeze_layers_config = self.config['train']['SiteSupConTrainer']['unfreeze_esm_layers']
        esm_module = self.model.esm_model.model
        for param in esm_module.parameters():
            param.requires_grad = False
        if unfreeze_layers_config == "all":
            for param in esm_module.parameters():
                param.requires_grad = True
        elif isinstance(unfreeze_layers_config, list) and unfreeze_layers_config:
            for layer_idx in unfreeze_layers_config:
                if 0 <= layer_idx - 1 < len(esm_module.layers):
                    for param in esm_module.layers[layer_idx - 1].parameters():
                        param.requires_grad = True
            for param in esm_module.emb_layer_norm_after.parameters():
                param.requires_grad = True
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]


        # ========================================data========================================
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
            num_samples=None,           # Deprecated arguments
            esm_model_name='esm2_t33_650M_UR50D',
            msa_model_name='esm_msa1b_t12_100M_UR50S',
            num_workers=self.config['common']['num_workers'],
            preprocessing=None,         # Deprecated arguments
            preprocessing_dir=None,     # Deprecated arguments
            filter_incomplete_ec=True,
        )
        self.batch_sampler = GroupedBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.config['train']['SiteSupConTrainer']['batch_size'], # 这里的 batch_size 是 Protein 数量
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler,
            num_workers=self.config['train']['SiteSupConTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=True,
            prefetch_factor=2
        )
        
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
            filter_incomplete_ec=False,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config['train']['SiteSupConTrainer']['batch_size'],
            shuffle=False,
            num_workers=self.config['train']['SiteSupConTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=True
        )
        print(f"[SiteSupConTrainer] Data Ready. Train Batches per Epoch: ~{len(self.batch_sampler)}")


        # ========================================optimizer, scheduler, loss function========================================
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(self.config['train']['SiteSupConTrainer']['learning_rate']),
            weight_decay=float(1e-2),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        steps_per_epoch = len(self.train_loader)
        total_training_steps = steps_per_epoch * self.config['train']['SiteSupConTrainer']['epoch']
        warmup_steps = int(steps_per_epoch * 5)  # 预热5个epoch
        self.scheduler = get_step_based_scheduler(
            self.optimizer,
            scheduler_type='cosine',
            total_steps=total_training_steps,
            warmup_steps=warmup_steps,
            min_lr=1e-6
        )
        self.loss_fn = create_loss_function(
            loss_type=self.config['train']['SiteSupConTrainer']['loss_type'],
            class_weights=self.config['train']['SiteSupConTrainer']['class_weights'],
            label_smoothing=self.config['train']['SiteSupConTrainer']['label_smoothing']
        ).to(self.device)


        # ========================================optimization========================================
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception as e:
                print(f"Optimization setup warning: {e}")



    def train(self):
        """Main training loop with proper epoch management."""
        print("Starting training...")
        print(f"Training for {self.config['train']['SiteSupConTrainer']['epoch']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['train']['SiteSupConTrainer']['epoch']):
            self.current_epoch = epoch  # Update current epoch at the beginning of each iteration
            
            train_info = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            print(f"Epoch {epoch+1}: Train Loss {train_info['loss']:.4f} (CE: {train_info['ce']:.4f}, Sup: {train_info['sup']:.4f})")
            metric_name = 'mcc' if self.config['model']['ResidueEncoderGamma']['num_class'] == 2 else 'multi_class_mcc'
            current_metric = val_metrics[metric_name]
            is_best = current_metric > self.best_val_metric

            if is_best:
                self.best_val_metric = current_metric
                print(f"New best model saved! MCC: {self.best_val_metric:.4f}")
            self.save_checkpoint(is_best=is_best)
            
        # Training completed
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation metric: {self.best_val_metric:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(filename="final_checkpoint.pt")



    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_sup = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            try:
                if batch is None:
                    continue
                batch = move_batch_to_device(batch, self.device)
                with autocast(enabled=self.use_amp, dtype=self.dtype):
                    # 1. 前向传播
                    logits, z, valid_mask = self.model(batch, return_z=True)

                    flat_targets = batch['targets'].long()          # [N_valid]
                    flat_ec_ints = batch['residue_ec_ints'].long()  # [N_valid]

                    # 2. Loss 1: CE (保持不变，用于区分背景和前景)
                    loss_ce = self.loss_fn(logits, flat_targets)

                    # 3. Loss 2: SupCon (修正：仅在活性位点之间进行对比)
                    lambda_ce = self.config['train']['SiteSupConTrainer']['supcon']['lambda_ce']
                    lambda_supcon = self.config['train']['SiteSupConTrainer']['supcon']['lambda_supcon']
                    temperature = self.config['train']['SiteSupConTrainer']['supcon']['temperature']
                    hard_pos_weight = self.config['train']['SiteSupConTrainer']['supcon']['hard_pos_weight']
                    hard_neg_weight = self.config['train']['SiteSupConTrainer']['supcon']['hard_neg_weight']

                    loss_supcon = torch.tensor(0.0, device=self.device)
                    
                    # === 修改开始 ===
                    # 关键修复：仅提取 Label > 0 的残基参与 SupCon 计算
                    # 这样可以排除 95% 的背景噪声，避免分母爆炸，并聚焦于活性位点的细粒度聚类
                    active_mask_indices = torch.nonzero(flat_targets > 0, as_tuple=True)[0]
                    
                    # 只有当 batch 中至少有 2 个活性位点时才计算 SupCon
                    if active_mask_indices.numel() > 1:
                        # 提取活性子集特征和标签
                        z_active = z[active_mask_indices]                 # [N_active, D]
                        labels_active = flat_targets[active_mask_indices] # [N_active]
                        ec_active = flat_ec_ints[active_mask_indices]     # [N_active]
                        
                        # 在活性子集上计算相似度矩阵 [N_active, N_active]
                        z_norm = F.normalize(z_active, dim=-1, p=2)
                        sim = torch.matmul(z_norm, z_norm.T) / temperature 

                        # 数值稳定
                        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
                        sim = sim - sim_max.detach()

                        # 构造 Mask (都在活性子集范围内)
                        N_active = z_active.shape[0]
                        # 广播生成矩阵
                        same_type = (labels_active.view(-1, 1) == labels_active.view(1, -1))
                        same_ec = (ec_active.view(-1, 1) == ec_active.view(1, -1))
                        
                        # 自身 Mask
                        self_mask = torch.eye(N_active, device=self.device, dtype=torch.bool)
                        
                        # Positive 定义: 同类型 (且不是自己)
                        # 注: 由于已经过滤了 targets>0，这里的 same_type 自动隐含了 active 条件
                        pos_all = same_type & (~self_mask)
                        
                        # Hard Positive: 同类型 且 同EC
                        pos_hard = pos_all & same_ec
                        
                        # 权重矩阵
                        pos_weight = torch.ones_like(sim, device=self.device)
                        pos_weight[pos_hard] = hard_pos_weight
                        
                        # Negative 定义: 
                        # Hard Negative: 同EC 但 不同类型 (最难区分的样本)
                        # Easy Negative: 其他所有 (不同EC 或 不同类型)
                        neg_hard = same_ec & (~same_type)
                        
                        neg_weight = torch.ones_like(sim, device=self.device)
                        neg_weight[neg_hard] = hard_neg_weight # 加大 Hard Negative 在分母的权重

                        # 计算 Log Prob
                        exp_sim = torch.exp(sim)
                        # 排除自己 (self_mask 位置设为0)，并应用负样本权重
                        exp_sim = exp_sim * (~self_mask).float() * neg_weight
                        
                        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True) + 1e-6
                        log_prob = sim - torch.log(exp_sim_sum)

                        # 聚合 Loss
                        # L_i = - (1 / |P_i|) * sum_{p in P_i} log_prob[i, p]
                        pos_mask_float = pos_all.float()
                        weighted_log_prob = pos_mask_float * pos_weight * log_prob
                        
                        # 分母归一化项 (权重的和)
                        normalizer = (pos_mask_float * pos_weight).sum(dim=1)
                        
                        # 仅对存在正样本的 anchor 计算 loss
                        valid_anchors = normalizer > 0
                        if valid_anchors.any():
                            loss_supcon = -(weighted_log_prob[valid_anchors].sum(dim=1) / normalizer[valid_anchors]).mean()
                    
                    # === 修改结束 ===

                    # 4. 总损失
                    loss = lambda_ce * loss_ce + lambda_supcon * loss_supcon

                # 5. 反向传播与优化
                if not self.check_loss_validity(loss):
                    self.optimizer.zero_grad()
                    continue

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['SiteSupConTrainer']['gradient_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['SiteSupConTrainer']['gradient_clip'])
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_ce += loss_ce.item()
                total_sup += loss_supcon.item() if isinstance(loss_supcon, torch.Tensor) else loss_supcon
                n_batches += 1
                pbar.set_postfix({
                    'L': loss.item(),
                    'CE': loss_ce.item(), 
                    'Sup': loss_supcon.item() if isinstance(loss_supcon, torch.Tensor) else 0.0,
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
            
            except Exception as e:
                uniprot_ids = batch.get(
                    'uniprot_ids',
                    ['N/A'] * self.config['train']['SiteSupConTrainer']['batch_size']
                )
                print(f"Training batch {batch_idx} failed and will be skipped. Error: {e}")
                print(f"Problematic UniProt IDs might be in: {uniprot_ids}")
                sys.stdout.flush()
                sys.stderr.flush()

        return {
            'loss': total_loss / max(1, n_batches),
            'ce': total_ce / max(1, n_batches),
            'sup': total_sup / max(1, n_batches)
        }


    @torch.no_grad()
    def validate_epoch(self, epoch) -> Dict[str, float]:
        """验证一个epoch，返回分类指标。"""
        self.model.eval()
        all_metrics = defaultdict(list)
        num_classes = self.config['model']['ResidueEncoderGamma']['num_class']
        
        pbar = tqdm(
            enumerate(self.valid_loader),
            total=len(self.valid_loader),
            desc=f"Epoch {epoch + 1}/{self.config['train']['SiteSupConTrainer']['epoch']} [VAL]",
            unit="batch"
        )
        
        try:
            for batch_idx, batch in pbar:
                if batch_idx >= len(self.valid_loader):
                    break
                if batch is None:
                    continue

                try:
                    batch = move_batch_to_device(batch, self.device)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        flat_logits, protein_mask = self.model(batch, return_z=False)   # 推理模式：return_z=False
                        flat_labels = batch['targets'].long()
                        assert flat_logits.size(0) == flat_labels.size(0), f"Mismatch: logits({flat_logits.size(0)}) vs targets({flat_labels.size(0)})"
                except Exception as e:
                    uniprot_ids = batch.get('uniprot_ids', ['N/A'])
                    print(f"Validation batch {batch_idx} failed. Error: {e}, UniProt IDs: {uniprot_ids}")
                    continue

                # Data Format Adaptation: 将flat格式转换为batch格式
                batch_size, max_len = protein_mask.shape
                labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
                logits = torch.zeros(batch_size, max_len, num_classes, device=self.device)
                labels[protein_mask.bool()] = flat_labels
                logits[protein_mask.bool()] = flat_logits
                valid_mask = protein_mask

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

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
        finally:
            if hasattr(pbar, 'close'):
                pbar.close()

        # 计算所有指标的平均值
        classification_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                classification_metrics[metric_name] = sum(values) / len(values)
        for metric_name, value in sorted(classification_metrics.items()):
            print(f"  {metric_name}: {value:.4f}")
        return classification_metrics



    def check_loss_validity(self, loss) -> bool:
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss detected: {loss.item()}")
            return False
        return True



    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model checkpoint
            filename: Optional checkpoint filename (defaults to epoch-based naming)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pt"      # Use 1-based epoch numbering for checkpoint filename
        filepath = os.path.join(self.config['common']['SiteSupCon_model_path'], filename)

        model_to_save = self.model

        
        checkpoint = {
            # Model state: core model parameters
            'model_state': {
                'state_dict': model_to_save.state_dict(),
                'model_class': model_to_save.__class__.__name__
            },
            
            # Optimizer state: optimization parameters and momentum
            'optimizer_state': {
                'state_dict': self.optimizer.state_dict(),
                'optimizer_class': self.optimizer.__class__.__name__
            },
            
            # Scheduler state: learning rate scheduling
            'scheduler_state': {
                'state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'scheduler_class': self.scheduler.__class__.__name__ if self.scheduler is not None else None
            },
            
            # Mixed precision state: gradient scaling for AMP
            'scaler_state': {
                'state_dict': self.scaler.state_dict() if self.scaler is not None else None,
                'enabled': self.use_amp
            },
            
            # Training state: epoch tracking and best metric
            'training_state': {
                'epoch': self.current_epoch,
                'best_val_metric': self.best_val_metric,
            },
            
            # Config state: preserve configuration
            'config_state': {
                'config': self.config
            },
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Modular checkpoint saved to {filepath}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['common']['SiteSupCon_model_path'], "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model checkpoint saved to {best_path}")



class SiteSupConAlphaTrainer:
    def __init__(self, config: Dict):

        self.config = config

        os.makedirs(self.config['common']['SiteSupCon_model_path'], exist_ok=True)
        self.device = torch.device(self.config['common']['device'])
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32
        self.scaler = GradScaler() if self.use_amp else None
        self.best_val_metric = -1.0
        self.current_epoch = 0

        # ========================================model========================================
        self.model = ResidueEncoderAlpha(self.config['model']['ResidueEncoderAlpha']).to(self.device)
        self.model.float()

        pretrained_dir_path = self.config['common']['ResidueEncoder_model_path']
        if pretrained_dir_path and os.path.isdir(pretrained_dir_path):
            try:
                pretrained_model_path = os.path.join(pretrained_dir_path, "best_model.pt")
                checkpoint = torch.load(pretrained_model_path, map_location=self.device)
                if 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']['state_dict']
                else:
                    state_dict = checkpoint
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if len(missing_keys) > 0:       print(f"  [Warning] Missing keys in backbone: {missing_keys[:5]} ... (Total: {len(missing_keys)})")
                if len(unexpected_keys) > 0:    print(f"  [Warning] Unexpected keys in checkpoint: {unexpected_keys[:5]} ... (Total: {len(unexpected_keys)})")
            except Exception as e:
                print(f"[SiteSupConTrainer] Error loading pretrained checkpoint: {e}")
        else:
            print("[SiteSupConTrainer] No 'pretrained_checkpoint_path' found in config or file does not exist.")
            print("[SiteSupConTrainer] Model will start with RANDOM initialization.")

        unfreeze_layers_config = self.config['train']['SiteSupConTrainer']['unfreeze_esm_layers']
        esm_module = self.model.esm_model.model
        for param in esm_module.parameters():
            param.requires_grad = False
        if unfreeze_layers_config == "all":
            for param in esm_module.parameters():
                param.requires_grad = True
        elif isinstance(unfreeze_layers_config, list) and unfreeze_layers_config:
            for layer_idx in unfreeze_layers_config:
                if 0 <= layer_idx - 1 < len(esm_module.layers):
                    for param in esm_module.layers[layer_idx - 1].parameters():
                        param.requires_grad = True
            for param in esm_module.emb_layer_norm_after.parameters():
                param.requires_grad = True
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]


        # ========================================data========================================
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
            num_samples=None,           # Deprecated arguments
            esm_model_name='esm2_t33_650M_UR50D',
            msa_model_name='esm_msa1b_t12_100M_UR50S',
            num_workers=self.config['common']['num_workers'],
            preprocessing=None,         # Deprecated arguments
            preprocessing_dir=None,     # Deprecated arguments
            filter_incomplete_ec=True,
        )
        self.batch_sampler = GroupedBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.config['train']['SiteSupConTrainer']['batch_size'], # 这里的 batch_size 是 Protein 数量
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler,
            num_workers=self.config['train']['SiteSupConTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=True,
            prefetch_factor=2
        )
        
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
            filter_incomplete_ec=False,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config['train']['SiteSupConTrainer']['batch_size'],
            shuffle=False,
            num_workers=self.config['train']['SiteSupConTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=True
        )
        print(f"[SiteSupConTrainer] Data Ready. Train Batches per Epoch: ~{len(self.batch_sampler)}")


        # ========================================optimizer, scheduler, loss function========================================
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(self.config['train']['SiteSupConTrainer']['learning_rate']),
            weight_decay=float(1e-2),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        steps_per_epoch = len(self.train_loader)
        total_training_steps = steps_per_epoch * self.config['train']['SiteSupConTrainer']['epoch']
        warmup_steps = int(steps_per_epoch * 5)  # 预热5个epoch
        self.scheduler = get_step_based_scheduler(
            self.optimizer,
            scheduler_type='cosine',
            total_steps=total_training_steps,
            warmup_steps=warmup_steps,
            min_lr=1e-6
        )
        self.loss_fn = create_loss_function(
            loss_type=self.config['train']['SiteSupConTrainer']['loss_type'],
            class_weights=self.config['train']['SiteSupConTrainer']['class_weights'],
            label_smoothing=self.config['train']['SiteSupConTrainer']['label_smoothing']
        ).to(self.device)


        # ========================================optimization========================================
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception as e:
                print(f"Optimization setup warning: {e}")



    def train(self):
        """Main training loop with proper epoch management."""
        print("Starting training...")
        print(f"Training for {self.config['train']['SiteSupConTrainer']['epoch']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['train']['SiteSupConTrainer']['epoch']):
            self.current_epoch = epoch  # Update current epoch at the beginning of each iteration
            
            train_info = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            print(f"Epoch {epoch+1}: Train Loss {train_info['loss']:.4f} (CE: {train_info['ce']:.4f}, Sup: {train_info['sup']:.4f})")
            metric_name = 'mcc' if self.config['model']['ResidueEncoderAlpha']['num_class'] == 2 else 'multi_class_mcc'
            current_metric = val_metrics[metric_name]
            is_best = current_metric > self.best_val_metric

            if is_best:
                self.best_val_metric = current_metric
                print(f"New best model saved! MCC: {self.best_val_metric:.4f}")
            self.save_checkpoint(is_best=is_best)
            
        # Training completed
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation metric: {self.best_val_metric:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(filename="final_checkpoint.pt")



    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_sup = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            try:
                if batch is None:
                    continue
                batch = move_batch_to_device(batch, self.device)
                with autocast(enabled=self.use_amp, dtype=self.dtype):
                    # 1. 前向传播
                    logits, z, valid_mask = self.model(batch, return_z=True)

                    flat_targets = batch['targets'].long()          # [N_valid]
                    flat_ec_ints = batch['residue_ec_ints'].long()  # [N_valid]

                    # 2. Loss 1: CE (保持不变，用于区分背景和前景)
                    loss_ce = self.loss_fn(logits, flat_targets)

                    # 3. Loss 2: SupCon (修正：仅在活性位点之间进行对比)
                    lambda_ce = self.config['train']['SiteSupConTrainer']['supcon']['lambda_ce']
                    lambda_supcon = self.config['train']['SiteSupConTrainer']['supcon']['lambda_supcon']
                    temperature = self.config['train']['SiteSupConTrainer']['supcon']['temperature']
                    hard_pos_weight = self.config['train']['SiteSupConTrainer']['supcon']['hard_pos_weight']
                    hard_neg_weight = self.config['train']['SiteSupConTrainer']['supcon']['hard_neg_weight']

                    loss_supcon = torch.tensor(0.0, device=self.device)
                    
                    # === 修改开始 ===
                    # 关键修复：仅提取 Label > 0 的残基参与 SupCon 计算
                    # 这样可以排除 95% 的背景噪声，避免分母爆炸，并聚焦于活性位点的细粒度聚类
                    active_mask_indices = torch.nonzero(flat_targets > 0, as_tuple=True)[0]
                    
                    # 只有当 batch 中至少有 2 个活性位点时才计算 SupCon
                    if active_mask_indices.numel() > 1:
                        # 提取活性子集特征和标签
                        z_active = z[active_mask_indices]                 # [N_active, D]
                        labels_active = flat_targets[active_mask_indices] # [N_active]
                        ec_active = flat_ec_ints[active_mask_indices]     # [N_active]
                        
                        # 在活性子集上计算相似度矩阵 [N_active, N_active]
                        z_norm = F.normalize(z_active, dim=-1, p=2)
                        sim = torch.matmul(z_norm, z_norm.T) / temperature 

                        # 数值稳定
                        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
                        sim = sim - sim_max.detach()

                        # 构造 Mask (都在活性子集范围内)
                        N_active = z_active.shape[0]
                        # 广播生成矩阵
                        same_type = (labels_active.view(-1, 1) == labels_active.view(1, -1))
                        same_ec = (ec_active.view(-1, 1) == ec_active.view(1, -1))
                        
                        # 自身 Mask
                        self_mask = torch.eye(N_active, device=self.device, dtype=torch.bool)
                        
                        # Positive 定义: 同类型 (且不是自己)
                        # 注: 由于已经过滤了 targets>0，这里的 same_type 自动隐含了 active 条件
                        pos_all = same_type & (~self_mask)
                        
                        # Hard Positive: 同类型 且 同EC
                        pos_hard = pos_all & same_ec
                        
                        # 权重矩阵
                        pos_weight = torch.ones_like(sim, device=self.device)
                        pos_weight[pos_hard] = hard_pos_weight
                        
                        # Negative 定义: 
                        # Hard Negative: 同EC 但 不同类型 (最难区分的样本)
                        # Easy Negative: 其他所有 (不同EC 或 不同类型)
                        neg_hard = same_ec & (~same_type)
                        
                        neg_weight = torch.ones_like(sim, device=self.device)
                        neg_weight[neg_hard] = hard_neg_weight # 加大 Hard Negative 在分母的权重

                        # 计算 Log Prob
                        exp_sim = torch.exp(sim)
                        # 排除自己 (self_mask 位置设为0)，并应用负样本权重
                        exp_sim = exp_sim * (~self_mask).float() * neg_weight
                        
                        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True) + 1e-6
                        log_prob = sim - torch.log(exp_sim_sum)

                        # 聚合 Loss
                        # L_i = - (1 / |P_i|) * sum_{p in P_i} log_prob[i, p]
                        pos_mask_float = pos_all.float()
                        weighted_log_prob = pos_mask_float * pos_weight * log_prob
                        
                        # 分母归一化项 (权重的和)
                        normalizer = (pos_mask_float * pos_weight).sum(dim=1)
                        
                        # 仅对存在正样本的 anchor 计算 loss
                        valid_anchors = normalizer > 0
                        if valid_anchors.any():
                            loss_supcon = -(weighted_log_prob[valid_anchors].sum(dim=1) / normalizer[valid_anchors]).mean()
                    
                    # === 修改结束 ===

                    # 4. 总损失
                    loss = lambda_ce * loss_ce + lambda_supcon * loss_supcon

                # 5. 反向传播与优化
                if not self.check_loss_validity(loss):
                    self.optimizer.zero_grad()
                    continue

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['SiteSupConTrainer']['gradient_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['SiteSupConTrainer']['gradient_clip'])
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_ce += loss_ce.item()
                total_sup += loss_supcon.item() if isinstance(loss_supcon, torch.Tensor) else loss_supcon
                n_batches += 1
                pbar.set_postfix({
                    'L': loss.item(),
                    'CE': loss_ce.item(), 
                    'Sup': loss_supcon.item() if isinstance(loss_supcon, torch.Tensor) else 0.0,
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
            
            except Exception as e:
                uniprot_ids = batch.get(
                    'uniprot_ids',
                    ['N/A'] * self.config['train']['SiteSupConTrainer']['batch_size']
                )
                print(f"Training batch {batch_idx} failed and will be skipped. Error: {e}")
                print(f"Problematic UniProt IDs might be in: {uniprot_ids}")
                sys.stdout.flush()
                sys.stderr.flush()

        return {
            'loss': total_loss / max(1, n_batches),
            'ce': total_ce / max(1, n_batches),
            'sup': total_sup / max(1, n_batches)
        }


    @torch.no_grad()
    def validate_epoch(self, epoch) -> Dict[str, float]:
        """验证一个epoch，返回分类指标。"""
        self.model.eval()
        all_metrics = defaultdict(list)
        num_classes = self.config['model']['ResidueEncoderAlpha']['num_class']
        
        pbar = tqdm(
            enumerate(self.valid_loader),
            total=len(self.valid_loader),
            desc=f"Epoch {epoch + 1}/{self.config['train']['SiteSupConTrainer']['epoch']} [VAL]",
            unit="batch"
        )
        
        try:
            for batch_idx, batch in pbar:
                if batch_idx >= len(self.valid_loader):
                    break
                if batch is None:
                    continue

                try:
                    batch = move_batch_to_device(batch, self.device)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        flat_logits, protein_mask = self.model(batch, return_z=False)   # 推理模式：return_z=False
                        flat_labels = batch['targets'].long()
                        assert flat_logits.size(0) == flat_labels.size(0), f"Mismatch: logits({flat_logits.size(0)}) vs targets({flat_labels.size(0)})"
                except Exception as e:
                    uniprot_ids = batch.get('uniprot_ids', ['N/A'])
                    print(f"Validation batch {batch_idx} failed. Error: {e}, UniProt IDs: {uniprot_ids}")
                    continue

                # Data Format Adaptation: 将flat格式转换为batch格式
                batch_size, max_len = protein_mask.shape
                labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
                logits = torch.zeros(batch_size, max_len, num_classes, device=self.device)
                labels[protein_mask.bool()] = flat_labels
                logits[protein_mask.bool()] = flat_logits
                valid_mask = protein_mask

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

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
        finally:
            if hasattr(pbar, 'close'):
                pbar.close()

        # 计算所有指标的平均值
        classification_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                classification_metrics[metric_name] = sum(values) / len(values)
        for metric_name, value in sorted(classification_metrics.items()):
            print(f"  {metric_name}: {value:.4f}")
        return classification_metrics



    def check_loss_validity(self, loss) -> bool:
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss detected: {loss.item()}")
            return False
        return True



    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model checkpoint
            filename: Optional checkpoint filename (defaults to epoch-based naming)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pt"      # Use 1-based epoch numbering for checkpoint filename
        filepath = os.path.join(self.config['common']['SiteSupCon_model_path'], filename)

        model_to_save = self.model

        
        checkpoint = {
            # Model state: core model parameters
            'model_state': {
                'state_dict': model_to_save.state_dict(),
                'model_class': model_to_save.__class__.__name__
            },
            
            # Optimizer state: optimization parameters and momentum
            'optimizer_state': {
                'state_dict': self.optimizer.state_dict(),
                'optimizer_class': self.optimizer.__class__.__name__
            },
            
            # Scheduler state: learning rate scheduling
            'scheduler_state': {
                'state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'scheduler_class': self.scheduler.__class__.__name__ if self.scheduler is not None else None
            },
            
            # Mixed precision state: gradient scaling for AMP
            'scaler_state': {
                'state_dict': self.scaler.state_dict() if self.scaler is not None else None,
                'enabled': self.use_amp
            },
            
            # Training state: epoch tracking and best metric
            'training_state': {
                'epoch': self.current_epoch,
                'best_val_metric': self.best_val_metric,
            },
            
            # Config state: preserve configuration
            'config_state': {
                'config': self.config
            },
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Modular checkpoint saved to {filepath}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['common']['SiteSupCon_model_path'], "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model checkpoint saved to {best_path}")



class SiteSupConBetaTrainer:
    def __init__(self, config: Dict):

        self.config = config

        os.makedirs(self.config['common']['SiteSupCon_model_path'], exist_ok=True)
        self.device = torch.device(self.config['common']['device'])
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32
        self.scaler = GradScaler() if self.use_amp else None
        self.best_val_metric = -1.0
        self.current_epoch = 0

        # ========================================model========================================
        self.model = ResidueEncoderBeta(self.config['model']['ResidueEncoderBeta']).to(self.device)
        self.model.float()

        pretrained_dir_path = self.config['common']['ResidueEncoder_model_path']
        if pretrained_dir_path and os.path.isdir(pretrained_dir_path):
            try:
                pretrained_model_path = os.path.join(pretrained_dir_path, "best_model.pt")
                checkpoint = torch.load(pretrained_model_path, map_location=self.device)
                if 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']['state_dict']
                else:
                    state_dict = checkpoint
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if len(missing_keys) > 0:       print(f"  [Warning] Missing keys in backbone: {missing_keys[:5]} ... (Total: {len(missing_keys)})")
                if len(unexpected_keys) > 0:    print(f"  [Warning] Unexpected keys in checkpoint: {unexpected_keys[:5]} ... (Total: {len(unexpected_keys)})")
            except Exception as e:
                print(f"[SiteSupConTrainer] Error loading pretrained checkpoint: {e}")
        else:
            print("[SiteSupConTrainer] No 'pretrained_checkpoint_path' found in config or file does not exist.")
            print("[SiteSupConTrainer] Model will start with RANDOM initialization.")

        unfreeze_layers_config = self.config['train']['SiteSupConTrainer']['unfreeze_esm_layers']
        esm_module = self.model.esm_model.model
        for param in esm_module.parameters():
            param.requires_grad = False
        if unfreeze_layers_config == "all":
            for param in esm_module.parameters():
                param.requires_grad = True
        elif isinstance(unfreeze_layers_config, list) and unfreeze_layers_config:
            for layer_idx in unfreeze_layers_config:
                if 0 <= layer_idx - 1 < len(esm_module.layers):
                    for param in esm_module.layers[layer_idx - 1].parameters():
                        param.requires_grad = True
            for param in esm_module.emb_layer_norm_after.parameters():
                param.requires_grad = True
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]


        # ========================================data========================================
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
            num_samples=None,           # Deprecated arguments
            esm_model_name='esm2_t33_650M_UR50D',
            msa_model_name='esm_msa1b_t12_100M_UR50S',
            num_workers=self.config['common']['num_workers'],
            preprocessing=None,         # Deprecated arguments
            preprocessing_dir=None,     # Deprecated arguments
            filter_incomplete_ec=True,
        )
        self.batch_sampler = GroupedBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.config['train']['SiteSupConTrainer']['batch_size'], # 这里的 batch_size 是 Protein 数量
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler,
            num_workers=self.config['train']['SiteSupConTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=True,
            prefetch_factor=2
        )
        
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
            filter_incomplete_ec=False,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config['train']['SiteSupConTrainer']['batch_size'],
            shuffle=False,
            num_workers=self.config['train']['SiteSupConTrainer']['num_workers'],
            collate_fn=collate_enzyme_batch,
            pin_memory=True
        )
        print(f"[SiteSupConTrainer] Data Ready. Train Batches per Epoch: ~{len(self.batch_sampler)}")


        # ========================================optimizer, scheduler, loss function========================================
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(self.config['train']['SiteSupConTrainer']['learning_rate']),
            weight_decay=float(1e-2),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        steps_per_epoch = len(self.train_loader)
        total_training_steps = steps_per_epoch * self.config['train']['SiteSupConTrainer']['epoch']
        warmup_steps = int(steps_per_epoch * 5)  # 预热5个epoch
        self.scheduler = get_step_based_scheduler(
            self.optimizer,
            scheduler_type='cosine',
            total_steps=total_training_steps,
            warmup_steps=warmup_steps,
            min_lr=1e-6
        )
        self.loss_fn = create_loss_function(
            loss_type=self.config['train']['SiteSupConTrainer']['loss_type'],
            class_weights=self.config['train']['SiteSupConTrainer']['class_weights'],
            label_smoothing=self.config['train']['SiteSupConTrainer']['label_smoothing']
        ).to(self.device)


        # ========================================optimization========================================
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception as e:
                print(f"Optimization setup warning: {e}")



    def train(self):
        """Main training loop with proper epoch management."""
        print("Starting training...")
        print(f"Training for {self.config['train']['SiteSupConTrainer']['epoch']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['train']['SiteSupConTrainer']['epoch']):
            self.current_epoch = epoch  # Update current epoch at the beginning of each iteration
            
            train_info = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            print(f"Epoch {epoch+1}: Train Loss {train_info['loss']:.4f} (CE: {train_info['ce']:.4f}, Sup: {train_info['sup']:.4f})")
            metric_name = 'mcc' if self.config['model']['ResidueEncoderBeta']['num_class'] == 2 else 'multi_class_mcc'
            current_metric = val_metrics[metric_name]
            is_best = current_metric > self.best_val_metric

            if is_best:
                self.best_val_metric = current_metric
                print(f"New best model saved! MCC: {self.best_val_metric:.4f}")
            self.save_checkpoint(is_best=is_best)
            
        # Training completed
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation metric: {self.best_val_metric:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(filename="final_checkpoint.pt")



    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_sup = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            try:
                if batch is None:
                    continue
                batch = move_batch_to_device(batch, self.device)
                with autocast(enabled=self.use_amp, dtype=self.dtype):
                    # 1. 前向传播
                    logits, z, valid_mask = self.model(batch, return_z=True)

                    flat_targets = batch['targets'].long()          # [N_valid]
                    flat_ec_ints = batch['residue_ec_ints'].long()  # [N_valid]

                    # 2. Loss 1: CE (保持不变，用于区分背景和前景)
                    loss_ce = self.loss_fn(logits, flat_targets)

                    # 3. Loss 2: SupCon (修正：仅在活性位点之间进行对比)
                    lambda_ce = self.config['train']['SiteSupConTrainer']['supcon']['lambda_ce']
                    lambda_supcon = self.config['train']['SiteSupConTrainer']['supcon']['lambda_supcon']
                    temperature = self.config['train']['SiteSupConTrainer']['supcon']['temperature']
                    hard_pos_weight = self.config['train']['SiteSupConTrainer']['supcon']['hard_pos_weight']
                    hard_neg_weight = self.config['train']['SiteSupConTrainer']['supcon']['hard_neg_weight']

                    loss_supcon = torch.tensor(0.0, device=self.device)
                    
                    # === 修改开始 ===
                    # 关键修复：仅提取 Label > 0 的残基参与 SupCon 计算
                    # 这样可以排除 95% 的背景噪声，避免分母爆炸，并聚焦于活性位点的细粒度聚类
                    active_mask_indices = torch.nonzero(flat_targets > 0, as_tuple=True)[0]
                    
                    # 只有当 batch 中至少有 2 个活性位点时才计算 SupCon
                    if active_mask_indices.numel() > 1:
                        # 提取活性子集特征和标签
                        z_active = z[active_mask_indices]                 # [N_active, D]
                        labels_active = flat_targets[active_mask_indices] # [N_active]
                        ec_active = flat_ec_ints[active_mask_indices]     # [N_active]
                        
                        # 在活性子集上计算相似度矩阵 [N_active, N_active]
                        z_norm = F.normalize(z_active, dim=-1, p=2)
                        sim = torch.matmul(z_norm, z_norm.T) / temperature 

                        # 数值稳定
                        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
                        sim = sim - sim_max.detach()

                        # 构造 Mask (都在活性子集范围内)
                        N_active = z_active.shape[0]
                        # 广播生成矩阵
                        same_type = (labels_active.view(-1, 1) == labels_active.view(1, -1))
                        same_ec = (ec_active.view(-1, 1) == ec_active.view(1, -1))
                        
                        # 自身 Mask
                        self_mask = torch.eye(N_active, device=self.device, dtype=torch.bool)
                        
                        # Positive 定义: 同类型 (且不是自己)
                        # 注: 由于已经过滤了 targets>0，这里的 same_type 自动隐含了 active 条件
                        pos_all = same_type & (~self_mask)
                        
                        # Hard Positive: 同类型 且 同EC
                        pos_hard = pos_all & same_ec
                        
                        # 权重矩阵
                        pos_weight = torch.ones_like(sim, device=self.device)
                        pos_weight[pos_hard] = hard_pos_weight
                        
                        # Negative 定义: 
                        # Hard Negative: 同EC 但 不同类型 (最难区分的样本)
                        # Easy Negative: 其他所有 (不同EC 或 不同类型)
                        neg_hard = same_ec & (~same_type)
                        
                        neg_weight = torch.ones_like(sim, device=self.device)
                        neg_weight[neg_hard] = hard_neg_weight # 加大 Hard Negative 在分母的权重

                        # 计算 Log Prob
                        exp_sim = torch.exp(sim)
                        # 排除自己 (self_mask 位置设为0)，并应用负样本权重
                        exp_sim = exp_sim * (~self_mask).float() * neg_weight
                        
                        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True) + 1e-6
                        log_prob = sim - torch.log(exp_sim_sum)

                        # 聚合 Loss
                        # L_i = - (1 / |P_i|) * sum_{p in P_i} log_prob[i, p]
                        pos_mask_float = pos_all.float()
                        weighted_log_prob = pos_mask_float * pos_weight * log_prob
                        
                        # 分母归一化项 (权重的和)
                        normalizer = (pos_mask_float * pos_weight).sum(dim=1)
                        
                        # 仅对存在正样本的 anchor 计算 loss
                        valid_anchors = normalizer > 0
                        if valid_anchors.any():
                            loss_supcon = -(weighted_log_prob[valid_anchors].sum(dim=1) / normalizer[valid_anchors]).mean()
                    
                    # === 修改结束 ===

                    # 4. 总损失
                    loss = lambda_ce * loss_ce + lambda_supcon * loss_supcon

                # 5. 反向传播与优化
                if not self.check_loss_validity(loss):
                    self.optimizer.zero_grad()
                    continue

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['SiteSupConTrainer']['gradient_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['SiteSupConTrainer']['gradient_clip'])
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_ce += loss_ce.item()
                total_sup += loss_supcon.item() if isinstance(loss_supcon, torch.Tensor) else loss_supcon
                n_batches += 1
                pbar.set_postfix({
                    'L': loss.item(),
                    'CE': loss_ce.item(), 
                    'Sup': loss_supcon.item() if isinstance(loss_supcon, torch.Tensor) else 0.0,
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
            
            except Exception as e:
                uniprot_ids = batch.get(
                    'uniprot_ids',
                    ['N/A'] * self.config['train']['SiteSupConTrainer']['batch_size']
                )
                print(f"Training batch {batch_idx} failed and will be skipped. Error: {e}")
                print(f"Problematic UniProt IDs might be in: {uniprot_ids}")
                sys.stdout.flush()
                sys.stderr.flush()

        return {
            'loss': total_loss / max(1, n_batches),
            'ce': total_ce / max(1, n_batches),
            'sup': total_sup / max(1, n_batches)
        }


    @torch.no_grad()
    def validate_epoch(self, epoch) -> Dict[str, float]:
        """验证一个epoch，返回分类指标。"""
        self.model.eval()
        all_metrics = defaultdict(list)
        num_classes = self.config['model']['ResidueEncoderBeta']['num_class']
        
        pbar = tqdm(
            enumerate(self.valid_loader),
            total=len(self.valid_loader),
            desc=f"Epoch {epoch + 1}/{self.config['train']['SiteSupConTrainer']['epoch']} [VAL]",
            unit="batch"
        )
        
        try:
            for batch_idx, batch in pbar:
                if batch_idx >= len(self.valid_loader):
                    break
                if batch is None:
                    continue

                try:
                    batch = move_batch_to_device(batch, self.device)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        flat_logits, protein_mask = self.model(batch, return_z=False)   # 推理模式：return_z=False
                        flat_labels = batch['targets'].long()
                        assert flat_logits.size(0) == flat_labels.size(0), f"Mismatch: logits({flat_logits.size(0)}) vs targets({flat_labels.size(0)})"
                except Exception as e:
                    uniprot_ids = batch.get('uniprot_ids', ['N/A'])
                    print(f"Validation batch {batch_idx} failed. Error: {e}, UniProt IDs: {uniprot_ids}")
                    continue

                # Data Format Adaptation: 将flat格式转换为batch格式
                batch_size, max_len = protein_mask.shape
                labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
                logits = torch.zeros(batch_size, max_len, num_classes, device=self.device)
                labels[protein_mask.bool()] = flat_labels
                logits[protein_mask.bool()] = flat_logits
                valid_mask = protein_mask

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

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
        finally:
            if hasattr(pbar, 'close'):
                pbar.close()

        # 计算所有指标的平均值
        classification_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                classification_metrics[metric_name] = sum(values) / len(values)
        for metric_name, value in sorted(classification_metrics.items()):
            print(f"  {metric_name}: {value:.4f}")
        return classification_metrics



    def check_loss_validity(self, loss) -> bool:
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss detected: {loss.item()}")
            return False
        return True



    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model checkpoint
            filename: Optional checkpoint filename (defaults to epoch-based naming)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pt"      # Use 1-based epoch numbering for checkpoint filename
        filepath = os.path.join(self.config['common']['SiteSupCon_model_path'], filename)

        model_to_save = self.model

        
        checkpoint = {
            # Model state: core model parameters
            'model_state': {
                'state_dict': model_to_save.state_dict(),
                'model_class': model_to_save.__class__.__name__
            },
            
            # Optimizer state: optimization parameters and momentum
            'optimizer_state': {
                'state_dict': self.optimizer.state_dict(),
                'optimizer_class': self.optimizer.__class__.__name__
            },
            
            # Scheduler state: learning rate scheduling
            'scheduler_state': {
                'state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'scheduler_class': self.scheduler.__class__.__name__ if self.scheduler is not None else None
            },
            
            # Mixed precision state: gradient scaling for AMP
            'scaler_state': {
                'state_dict': self.scaler.state_dict() if self.scaler is not None else None,
                'enabled': self.use_amp
            },
            
            # Training state: epoch tracking and best metric
            'training_state': {
                'epoch': self.current_epoch,
                'best_val_metric': self.best_val_metric,
            },
            
            # Config state: preserve configuration
            'config_state': {
                'config': self.config
            },
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Modular checkpoint saved to {filepath}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['common']['SiteSupCon_model_path'], "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model checkpoint saved to {best_path}")



class ActiveSiteEmbeddingNetTrainer:
    def __init__(
            self, 
            config: dict, 
            model: nn.Module, 
            train_residue_features: Dict[str, torch.Tensor],
            valid_residue_features: Dict[str, torch.Tensor],
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame
        ):
        """
        Args:
            config: 总配置文件
            model: ActiveSiteEmbeddingNet模型实例
            train_residue_features: 训练集残基特征的字典，需要注意的是train_residue_features可以是
                * 仅经过模块一处理的特征，
                * 也可以是经过模块一和模块二ECEmbeddingNet处理后的特征。
                {
                    uniprot_id_1: 二维张量 [L, D],  # 残基顺序严格按照PDB文件的残基顺序
                    uniprot_id_2: 二维张量 [L, D],
                }
            valid_residue_features: 验证集残基特征的字典，格式同上
            train_df: 已经在 ClusterSiteTrainer 中预处理好的训练集 DataFrame
            valid_df: 已经在 ClusterSiteTrainer 中预处理好的验证集 DataFrame
        """
        self.config = config
        
        self.device = torch.device(self.config['common']['device'])
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32

        assert model is not None, "ActiveSiteEmbeddingNetTrainer requires an injected ActiveSiteEmbeddingNet model."
        self.model = model.to(self.device)
        self.model.float()
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['train']['ActiveSiteEmbeddingNetTrainer']['learning_rate'], 
            betas=(0.9, 0.999)
        )
        self.scaler = GradScaler() if self.use_amp else None
        self.criterion = SupConHardLoss
        self.best_loss = float('inf')
        self.best_site_mcc = -1.0

        # =========================================== Top@Data Pools ============================
        (self.train_id_ec, 
        _, 
        _, 
        self.train_labels, 
        self.train_flat_features_initial, 
        self.train_ec_pools_indexed, 
        _, 
        self.train_uid_to_offset
        ) = build_data_pools(dataframe=train_df, 
                             embedding_dict=train_residue_features,
                             num_workers=self.config['common']['num_workers'], 
                             ec_level=self.config['common']['ec_level'],
                             cluster_mode=self.config['common']['cluster_mode']
        )
        
        (self.valid_id_ec, 
        _, 
        _, 
        self.valid_labels,
        self.valid_flat_features_initial, 
        _,
        _, 
        self.valid_uid_to_offset
        ) = build_data_pools(dataframe=valid_df, 
                             embedding_dict=valid_residue_features, 
                             num_workers=self.config['common']['num_workers'], 
                             ec_level=self.config['common']['ec_level'],
                             cluster_mode=self.config['common']['cluster_mode'])
        # =========================================== Bottom@Data Pools ==========================


        print("Initializing dataset and dataloader for active site training...")
        dataset = MultiPosNeg_dataset_with_mine_SITE(
            n_pos=self.config['train']['ActiveSiteEmbeddingNetTrainer']['n_pos_site'],
            n_neg=self.config['train']['ActiveSiteEmbeddingNetTrainer']['n_neg_site'],
            residue_features_dict=train_residue_features,
            dataframe=train_df,
            num_workers=self.config['train']['ActiveSiteEmbeddingNetTrainer']['num_workers'],
            hard_neg_ratio=self.config['train']['ActiveSiteEmbeddingNetTrainer']['hard_neg_ratio'],
            cluster_mode=self.config['common']['cluster_mode'],
            ec_level=self.config['common']['ec_level']
        )
        
        dataloader_params = {
            'batch_size': self.config['train']['ActiveSiteEmbeddingNetTrainer']['batch_size'],
            'shuffle': self.config['train']['ActiveSiteEmbeddingNetTrainer']['shuffle'],
            'num_workers': self.config['train']['ActiveSiteEmbeddingNetTrainer']['num_workers'],
            'pin_memory': True,
        }
        self.train_loader = DataLoader(dataset, **dataloader_params)



    def train(self):
        model_save_path = self.config['common']['ActiveSiteEmbeddingNet_model_path']
        if self.config['common']['ec_level'] != 4:
            model_save_path = model_save_path.rstrip('/') + f"_lvl{self.config['common']['ec_level']}"
        os.makedirs(model_save_path, exist_ok=True)
        
        pbar = tqdm(range(1, self.config['train']['ActiveSiteEmbeddingNetTrainer']['epoch'] + 1), desc="ActiveSiteEmbeddingNet Training")
        for epoch in pbar:
            if (epoch % self.config['train']['ActiveSiteEmbeddingNetTrainer']['adaptive_rate'] == 0) or (epoch == self.config['train']['ActiveSiteEmbeddingNetTrainer']['epoch']):
                checkpoint_path = os.path.join(model_save_path, f'checkpoint_epoch_{epoch}_site.pt')
                torch.save(self.model.state_dict(), checkpoint_path)

            train_loss = self.train_epoch(epoch)
            metrics = self.valid_epoch(epoch)
            site_mcc = metrics['site_mcc']

            pbar.set_postfix({
                'loss': f'{train_loss:.4f}', 
                'val_mcc': f'{site_mcc:.4f}',
                'best_mcc': f'{self.best_site_mcc:.4f}'
            })

            if site_mcc > self.best_site_mcc:
                self.best_site_mcc = site_mcc
                torch.save(self.model.state_dict(), os.path.join(model_save_path, 'best_model_site.pt'))



    def train_epoch(self, epoch):
        """
        Args:
            epoch (int): 当前的周期数。

        Returns:
            float: 该周期在所有批次上的平均损失。
        """
        self.model.train()
        total_loss = 0.
        
        for batch_idx, data in enumerate(self.train_loader):
            try:
                data = data.to(device=self.device)  # 只将数据移动到设备，精度转换完全交给 autocast 管理。
                self.optimizer.zero_grad()
                with autocast(enabled=self.use_amp, dtype=self.dtype):
                    model_emb = self.model(data)
                    loss = self.criterion(model_emb, 
                                        temp=self.config['train']['ActiveSiteEmbeddingNetTrainer']['temp'], 
                                        n_pos=self.config['train']['ActiveSiteEmbeddingNetTrainer']['n_pos_site'])
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error in batch {batch_idx} during ActiveSiteEmbeddingNetTrainer training epoch {epoch}: {e}")
                self.optimizer.zero_grad()
                continue
        return total_loss / (batch_idx + 1)


    @torch.no_grad()
    def valid_epoch(self, epoch):
        self.model.eval()
        metrics = {}

        # residue-level
        reference_residue_embs = encode_residue_feature(flat_features=self.train_flat_features_initial, model=self.model, batch_size=6000000, device=self.device, use_amp=self.use_amp, dtype=self.dtype)
        query_residue_embs = encode_residue_feature(flat_features=self.valid_flat_features_initial, model=self.model, batch_size=6000000, device=self.device, use_amp=self.use_amp, dtype=self.dtype)

        site_metrics_dict, _ = evaluate_site_type_prediction(
            reference_residue_features=reference_residue_embs,
            query_residue_features=query_residue_embs,
            ec_site_pools_indexed_reference=self.train_ec_pools_indexed,
            id_ec_query=self.valid_id_ec,
            labels_dict_query=self.valid_labels,
            uid_to_offset_query=self.valid_uid_to_offset,
            device=self.device,
        )
        
        if 'multi_class_mcc' in site_metrics_dict and site_metrics_dict['multi_class_mcc']:
            metrics['site_mcc'] = np.mean(site_metrics_dict['multi_class_mcc'])
        else:
            raise RuntimeError("Site multi-class MCC calculation failed during validation.")
        
        self.model.train()
        return metrics



class ActiveSiteEmbeddingNetBinaryTrainer(ActiveSiteEmbeddingNetTrainer):
    @torch.no_grad()
    def valid_epoch(self, epoch):
        self.model.eval()
        metrics = {}

        reference_residue_embs = encode_residue_feature(
            flat_features=self.train_flat_features_initial, model=self.model, 
            batch_size=6000000, device=self.device, use_amp=self.use_amp, dtype=self.dtype
        )
        query_residue_embs = encode_residue_feature(
            flat_features=self.valid_flat_features_initial, model=self.model, 
            batch_size=6000000, device=self.device, use_amp=self.use_amp, dtype=self.dtype
        )

        # Use the binary evaluation function
        site_metrics_dict, _ = evaluate_site_type_prediction_binary(
            reference_residue_features=reference_residue_embs,
            query_residue_features=query_residue_embs,
            ec_site_pools_indexed_reference=self.train_ec_pools_indexed,
            id_ec_query=self.valid_id_ec,
            labels_dict_query=self.valid_labels,
            uid_to_offset_query=self.valid_uid_to_offset,
            device=self.device,
        )
        
        if 'mcc' in site_metrics_dict and site_metrics_dict['mcc']:
            metrics['site_mcc'] = np.mean(site_metrics_dict['mcc'])
        else:
            raise RuntimeError("Site multi-class MCC calculation failed during validation.")
        
        self.model.train()
        return metrics



class ClusterSiteTrainer:
    """
    * mode
        初始化一次ClusterSiteTrainer，只能调用train()一次，不能多次调用train()，只能传入一种switch，不能多次传入不同switch。
    * 关于Expected dataset format，参考EnzymeActiveSiteDataset3(EnzymeMSAStructureDataset)
    * 关于精度问题
        * config['common']['mixed_precision']==True时，使用混合精度训练
            * 模型权重：始终 float32，因此禁止出现 model.to(..., dtype=self.dtype)
            * 特征 / 大张量: float16
            * 前向计算：autocast(enabled=True, dtype=torch.float16)
        * config['common']['mixed_precision']==False时，使用单精度训练
            * 模型权重：float32
            * 特征 / 大张量: float32
            * 不用 autocast / GradScaler
    """
    def __init__(self, config: dict, args: Optional[argparse.Namespace] = None):
        """
        Args:
            config (dict): 总配置文件
        """
        self.config = config
        self.args = args

        self.device = torch.device(self.config['common']['device'])
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32   # self.dtype 控制“特征向量/计算 dtype”，不会再影响权重。
        
        print("Initializing ClusterSite composite model...")
        self.model = ClusterSite(self.config['model']).to(self.device)
        self.model.float()  # 显式保证权重为 float32

        """
        ClusterSiteTrainer类内共享特征，结构比如
            self.part1_residue_features: {
                "train": {
                    uniprot_id_1: 二维张量 [L, D],  # 残基顺序严格按照PDB文件的残基顺序
                    uniprot_id_2: 二维张量 [L, D],
                },
                "valid": {
                    uniprot_id_3: 二维张量 [L, D],
                    uniprot_id_4: 二维张量 [L, D],
                }
            }
        """
        self.part1_residue_features = {}    # 仅经过模块一编码器后的残基特征
        self.part1_protein_features = {}    # 仅经过模块一编码器后的蛋白质特征（平均池化）
        self.resume_checkpoint = None       # 初始化 resume_checkpoint 属性

        

    def train(self):
        switch = self.config['common']['switch']
        if switch == '1':
            print(">>> Starting Stage 1.1: ResidueEncoderGamma CE Training")
            trainer1 = ResidueEncoderGammaTrainer(self.config)
            if self.resume_checkpoint is not None:
                print(f"[ClusterSiteTrainer] Passing resume checkpoint to ResidueEncoderGammaTrainer: {self.resume_checkpoint}")
                trainer1.load_checkpoint(self.resume_checkpoint)
            trainer1.train()
            del trainer1
            gc.collect()
            torch.cuda.empty_cache()

            print(">>> Starting Stage 1.2: SiteSupCon Training")
            trainer2 = SiteSupConTrainer(self.config)
            trainer2.train()
            del trainer2
            gc.collect()
            torch.cuda.empty_cache()

        elif switch == '2':
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

            self.inference_1(['train', 'valid'])

            trainer3 = ActiveSiteEmbeddingNetTrainer(
                self.config,
                self.model.active_site_embedding_net,
                train_residue_features=self.part1_residue_features['train'],
                valid_residue_features=self.part1_residue_features['valid'],
                train_df=self.train_df,
                valid_df=self.valid_df
            )
            self.model.set_module_train_mode('active_site_embedding_net')
            trainer3.train()


    def inference_1(self, data_splits: List[str]):
        print(f"Running Inference on Part 1 for {data_splits} data.")

        if self.args and self.args.part1_model_path:
            model_path = self.args.part1_model_path
            print(f"Overriding Part 1 model path with command line argument: {model_path}")
        else:
            raise ValueError("Part 1 model path must be provided via command line argument '--part1_model_path'.")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Best model for Part 1 not found at {model_path}. Please run 'part-1' training first.")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']['state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        self.model.residue_encoder.load_state_dict(state_dict, strict=True)
        self.model.residue_encoder.to(self.device)
        self.model.residue_encoder.eval()

        for data_split in data_splits:
            print(f"Processing {data_split} data...")
            if data_split == 'train':
                dataset = self.train_dataset
            elif data_split == 'valid':
                dataset = self.valid_dataset
            else:
                raise ValueError(f"Invalid data_split: {data_split}.")

            dataloader = DataLoader(
                dataset,
                batch_size=48,
                shuffle=False,
                num_workers=self.config['common']['num_workers'],
                collate_fn=collate_enzyme_batch,
                pin_memory=False
            )

            residue_features_dict = {}
            protein_features_dict = {}
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Inference Residue Encoder for {data_split} set"):
                    if batch is None: continue
                    batch = move_batch_to_device(batch, self.device)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        protein_node_feature, protein_mask = self.model.residue_encoder(batch, return_features=True)
                        flat_features = protein_node_feature[protein_mask.bool()]
                    num_nodes_per_graph = batch['protein_graph'].num_residues.tolist()
                    residue_features_list = torch.split(flat_features, num_nodes_per_graph)
                    for i, uniprot_id in enumerate(batch['uniprot_ids']):
                        residues_tensor = residue_features_list[i].cpu()    # [L, D]
                        assert residues_tensor.size(0) == num_nodes_per_graph[i], f"残基数量不匹配: {residues_tensor.size(0)} vs {num_nodes_per_graph[i]} for protein {uniprot_id}"
                        residue_features_dict[uniprot_id] = residues_tensor.clone()
                        
                        pooled_embedding = residues_tensor.mean(dim=0)
                        protein_features_dict[uniprot_id] = pooled_embedding.to(self.dtype)
            self.part1_residue_features[data_split] = residue_features_dict
            self.part1_protein_features[data_split] = protein_features_dict
            print(f"Finished {data_split}: {len(residue_features_dict)} samples processed.")


    def load_checkpoint(self, checkpoint_path: str):
        self.resume_checkpoint = checkpoint_path




class ClusterSiteBinaryTrainer(ClusterSiteTrainer):
    def __init__(self, config: dict, args: Optional[argparse.Namespace] = None):
        super().__init__(config, args)
        # Initialize the Binary model
        self.model = ClusterSiteBinary(self.config['model']).to(self.device)
        self.model.float()

    def train(self):
        switch = self.config['common']['switch']
        if switch == '1':
            print(">>> Starting Stage 1.1: ResidueEncoderGammaBinary CE Training")
            # Use Binary Trainer
            trainer1 = ResidueEncoderGammaBinaryTrainer(self.config)
            if self.resume_checkpoint is not None:
                trainer1.load_checkpoint(self.resume_checkpoint)
            trainer1.train()
            del trainer1
            gc.collect()
            torch.cuda.empty_cache()

        elif switch == '2':
            # Use Binary Datasets
            self.train_dataset = EnzymeActiveSiteBinaryDataset3(
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

            self.valid_dataset = EnzymeActiveSiteBinaryDataset3(
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

            self.inference_1(['train', 'valid'])

            # Use Binary Embedding Net Trainer
            trainer3 = ActiveSiteEmbeddingNetBinaryTrainer(
                self.config,
                self.model.active_site_embedding_net,
                train_residue_features=self.part1_residue_features['train'],
                valid_residue_features=self.part1_residue_features['valid'],
                train_df=self.train_df,
                valid_df=self.valid_df
            )
            self.model.set_module_train_mode('active_site_embedding_net')
            trainer3.train()



class ClusterSiteAlphaTrainer:

    def __init__(self, config: dict, args: Optional[argparse.Namespace] = None):
        self.config = config
        self.args = args

        self.device = torch.device(self.config['common']['device'])
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32 
        
        print("Initializing ClusterSiteAlpha composite model...")
        self.model = ClusterSiteAlpha(self.config['model']).to(self.device)
        self.model.float()

        self.part1_residue_features = {}
        self.part1_protein_features = {}
        self.resume_checkpoint = None

    def train(self):
        switch = self.config['common']['switch']
        if switch == '1':
            print(">>> [ClusterSiteAlpha] Starting Stage 1.1: ResidueEncoderAlpha CE Training")
            # [CHANGE] 使用 ResidueEncoderAlphaTrainer
            trainer1 = ResidueEncoderAlphaTrainer(self.config)
            if self.resume_checkpoint is not None:
                print(f"[ClusterSiteAlpha] Passing resume checkpoint to ResidueEncoderAlphaTrainer: {self.resume_checkpoint}")
                trainer1.load_checkpoint(self.resume_checkpoint)
            trainer1.train()
            del trainer1
            gc.collect()
            torch.cuda.empty_cache()

            print(">>> [ClusterSiteAlpha] Starting Stage 1.2: SiteSupConBeta Training")
            # [CHANGE] 使用 SiteSupConAlphaTrainer
            trainer2 = SiteSupConAlphaTrainer(self.config)
            trainer2.train()
            del trainer2
            gc.collect()
            torch.cuda.empty_cache()

        elif switch == '2':
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

            # Run Inference on Part 1 (ResidueEncoderAlpha)
            self.inference_1(['train', 'valid'])

            # Train Part 2 (ActiveSiteEmbeddingNet)
            # 这里的逻辑与 ClusterSiteTrainer 完全一致，因为 Module 2 接受的是特征，不关心来源
            trainer3 = ActiveSiteEmbeddingNetTrainer(
                self.config,
                self.model.active_site_embedding_net,
                train_residue_features=self.part1_residue_features['train'],
                valid_residue_features=self.part1_residue_features['valid'],
                train_df=self.train_df,
                valid_df=self.valid_df
            )
            self.model.set_module_train_mode('active_site_embedding_net')
            trainer3.train()

    def inference_1(self, data_splits: List[str]):
        """
        使用训练好的 ResidueEncoderAlpha 提取特征。
        """
        print(f"Running Inference on Part 1 (Beta) for {data_splits} data.")

        if self.args and self.args.part1_model_path:
            model_path = self.args.part1_model_path
            print(f"Overriding Part 1 model path with command line argument: {model_path}")
        else:
            raise ValueError("Part 1 model path must be provided via command line argument '--part1_model_path'.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Best model for Part 1 not found at {model_path}. Please run 'part-1' training first.")
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']['state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # [CHANGE] 加载到 ResidueEncoderAlpha
        self.model.residue_encoder.load_state_dict(state_dict, strict=True)
        self.model.residue_encoder.to(self.device)
        self.model.residue_encoder.eval()

        for data_split in data_splits:
            print(f"Processing {data_split} data...")
            if data_split == 'train':
                dataset = self.train_dataset
            elif data_split == 'valid':
                dataset = self.valid_dataset
            else:
                raise ValueError(f"Invalid data_split: {data_split}.")

            dataloader = DataLoader(
                dataset,
                batch_size=48,
                shuffle=False,
                num_workers=self.config['common']['num_workers'],
                collate_fn=collate_enzyme_batch,
                pin_memory=False
            )

            residue_features_dict = {}
            protein_features_dict = {}
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Inference Residue Encoder Beta for {data_split} set"):
                    if batch is None: continue
                    batch = move_batch_to_device(batch, self.device)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        # [CHANGE] ResidueEncoderAlpha 的 forward
                        protein_node_feature, protein_mask = self.model.residue_encoder(batch, return_features=True)
                        flat_features = protein_node_feature[protein_mask.bool()]
                    
                    num_nodes_per_graph = batch['protein_graph'].num_residues.tolist()
                    residue_features_list = torch.split(flat_features, num_nodes_per_graph)
                    for i, uniprot_id in enumerate(batch['uniprot_ids']):
                        residues_tensor = residue_features_list[i].cpu()
                        residue_features_dict[uniprot_id] = residues_tensor.clone()
                        
                        pooled_embedding = residues_tensor.mean(dim=0)
                        protein_features_dict[uniprot_id] = pooled_embedding.to(self.dtype)
                                              
            self.part1_residue_features[data_split] = residue_features_dict
            self.part1_protein_features[data_split] = protein_features_dict
            print(f"Finished {data_split}: {len(residue_features_dict)} samples processed.")

    def load_checkpoint(self, checkpoint_path: str):
        self.resume_checkpoint = checkpoint_path




class ClusterSiteBetaTrainer:

    def __init__(self, config: dict, args: Optional[argparse.Namespace] = None):
        self.config = config
        self.args = args

        self.device = torch.device(self.config['common']['device'])
        self.use_amp = self.config['common']['mixed_precision']
        self.dtype = torch.float16 if self.use_amp else torch.float32 
        
        print("Initializing ClusterSiteBeta composite model...")
        self.model = ClusterSiteBeta(self.config['model']).to(self.device)
        self.model.float()

        self.part1_residue_features = {}
        self.part1_protein_features = {}
        self.resume_checkpoint = None

    def train(self):
        switch = self.config['common']['switch']
        if switch == '1':
            print(">>> [ClusterSiteBeta] Starting Stage 1.1: ResidueEncoderBeta CE Training")
            # [CHANGE] 使用 ResidueEncoderBetaTrainer
            trainer1 = ResidueEncoderBetaTrainer(self.config)
            if self.resume_checkpoint is not None:
                print(f"[ClusterSiteBeta] Passing resume checkpoint to ResidueEncoderBetaTrainer: {self.resume_checkpoint}")
                trainer1.load_checkpoint(self.resume_checkpoint)
            trainer1.train()
            del trainer1
            gc.collect()
            torch.cuda.empty_cache()

            print(">>> [ClusterSiteBeta] Starting Stage 1.2: SiteSupConBeta Training")
            # [CHANGE] 使用 SiteSupConBetaTrainer
            trainer2 = SiteSupConBetaTrainer(self.config)
            trainer2.train()
            del trainer2
            gc.collect()
            torch.cuda.empty_cache()

        elif switch == '2':
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

            # Run Inference on Part 1 (ResidueEncoderBeta)
            self.inference_1(['train', 'valid'])

            # Train Part 2 (ActiveSiteEmbeddingNet)
            # 这里的逻辑与 ClusterSiteTrainer 完全一致，因为 Module 2 接受的是特征，不关心来源
            trainer3 = ActiveSiteEmbeddingNetTrainer(
                self.config,
                self.model.active_site_embedding_net,
                train_residue_features=self.part1_residue_features['train'],
                valid_residue_features=self.part1_residue_features['valid'],
                train_df=self.train_df,
                valid_df=self.valid_df
            )
            self.model.set_module_train_mode('active_site_embedding_net')
            trainer3.train()

    def inference_1(self, data_splits: List[str]):
        """
        使用训练好的 ResidueEncoderBeta 提取特征。
        """
        print(f"Running Inference on Part 1 (Beta) for {data_splits} data.")

        if self.args and self.args.part1_model_path:
            model_path = self.args.part1_model_path
            print(f"Overriding Part 1 model path with command line argument: {model_path}")
        else:
            raise ValueError("Part 1 model path must be provided via command line argument '--part1_model_path'.")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Best model for Part 1 not found at {model_path}. Please run 'part-1' training first.")
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']['state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # [CHANGE] 加载到 ResidueEncoderBeta
        self.model.residue_encoder.load_state_dict(state_dict, strict=True)
        self.model.residue_encoder.to(self.device)
        self.model.residue_encoder.eval()

        for data_split in data_splits:
            print(f"Processing {data_split} data...")
            if data_split == 'train':
                dataset = self.train_dataset
            elif data_split == 'valid':
                dataset = self.valid_dataset
            else:
                raise ValueError(f"Invalid data_split: {data_split}.")

            dataloader = DataLoader(
                dataset,
                batch_size=48,
                shuffle=False,
                num_workers=self.config['common']['num_workers'],
                collate_fn=collate_enzyme_batch,
                pin_memory=False
            )

            residue_features_dict = {}
            protein_features_dict = {}
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Inference Residue Encoder Beta for {data_split} set"):
                    if batch is None: continue
                    batch = move_batch_to_device(batch, self.device)
                    with autocast(enabled=self.use_amp, dtype=self.dtype):
                        # [CHANGE] ResidueEncoderBeta 的 forward
                        protein_node_feature, protein_mask = self.model.residue_encoder(batch, return_features=True)
                        flat_features = protein_node_feature[protein_mask.bool()]
                    num_nodes_per_graph = batch['protein_graph'].num_residues.tolist()
                    residue_features_list = torch.split(flat_features, num_nodes_per_graph)
                    for i, uniprot_id in enumerate(batch['uniprot_ids']):
                        residues_tensor = residue_features_list[i].cpu()
                        residue_features_dict[uniprot_id] = residues_tensor.clone()
                        
                        pooled_embedding = residues_tensor.mean(dim=0)
                        protein_features_dict[uniprot_id] = pooled_embedding.to(self.dtype) 
            self.part1_residue_features[data_split] = residue_features_dict
            self.part1_protein_features[data_split] = protein_features_dict
            print(f"Finished {data_split}: {len(residue_features_dict)} samples processed.")

    def load_checkpoint(self, checkpoint_path: str):
        self.resume_checkpoint = checkpoint_path



def parse_args():
    parser = argparse.ArgumentParser(description="EvoSite Training Script")
    parser.add_argument("--config", type=str, required=True, help="Required. Path to training configuration file")
    parser.add_argument("--trainer", type=str, default="cluster_site", help="Required. Type of trainer to use")
    parser.add_argument("--part1-model-path", type=str, help="Optional: Path to override Part 1 model for inference.")


    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional. Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Optional. Local rank for distributed training"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Optional. Total number of GPUs for distributed training"
    )

    
    return parser.parse_args()



def main():
    """
    * 启用混合精度，请先按照如下进行修改
        /home/ubuntu/anaconda3/envs/EvoSite/lib/python3.9/site-packages/torchdrug/layers/conv.py-->class GeometricRelationalGraphConv-->def message_and_aggregate(self, graph, input)
        /home/ubuntu/anaconda3/envs/EvoSite/lib/python3.9/site-packages/torchdrug/layers/conv.py-->from torch.cuda.amp import autocast
    """
    # torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    
    trainer_type = args.trainer
    if trainer_type == "esm_msa_transformer_baseline_ablation":
        trainer = ESM_MSATransformer_AblationTrainer(config, local_rank=args.local_rank)
    elif trainer_type == "evosite_combined":
        trainer = EvoSiteCombinedTrainer(config, local_rank=args.local_rank)
    elif trainer_type == "evosite_beta":
        trainer = EvoSiteBetaTrainer(config, local_rank=args.local_rank)
    elif trainer_type == "evosite_gamma":
        trainer = EvoSiteGammaTrainer(config, local_rank=args.local_rank)
    elif trainer_type == "evosite_phignet":
        trainer = EvoSitePhiGnetTrainer(config, local_rank=args.local_rank)
    elif trainer_type == "evosite_betaphignet":
        trainer = EvoSiteBetaPhiGnetTrainer(config, local_rank=args.local_rank)
    elif trainer_type == "evosite_graphec":
        trainer = EvoSiteGraphECTrainer(config, local_rank=args.local_rank)
    elif trainer_type == "evosite_delta":
        trainer = EvoSiteDeltaTrainer(config, local_rank=args.local_rank)
    elif trainer_type == "residue_encoder_beta":
        trainer = ResidueEncoderBetaTrainer(config)
    elif trainer_type == "cluster_site":
        trainer = ClusterSiteTrainer(config=config, args=args)
    elif trainer_type == "evosite":
        trainer = EvoSiteTrainer(config=config, local_rank=args.local_rank)
    elif trainer_type == "site_supcon":
        trainer = SiteSupConTrainer(config=config)
    elif trainer_type == "cluster_site_alpha":
        trainer = ClusterSiteAlphaTrainer(config=config, args=args)
    elif trainer_type == "cluster_site_beta":
        trainer = ClusterSiteBetaTrainer(config=config, args=args)
    elif trainer_type == "cluster_site_binary":
        trainer = ClusterSiteBinaryTrainer(config=config, args=args)
    else:
        raise ValueError(f"Invalid trainer type: {trainer_type}")

    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.save_checkpoint(filename="interrupted_checkpoint.pt")
    except Exception as e:
        trainer.save_checkpoint(filename="error_checkpoint.pt")
        raise



if __name__ == "__main__":
    main()
