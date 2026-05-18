import torch
import torch.nn as nn
import esm



class EvoMSATransformer(nn.Module):
    """
    Uses MSA Transformer to extract evolutionary information from MSA files.
    """
    def __init__(
        self, 
        model_path="esm_msa1b_t12_100M_UR50S",
        device=None
    ):
        super(EvoMSATransformer, self).__init__()
        
        # ===========================================Load the MSA Transformer model===========================================
        try:
            print(f"Loading MSA Transformer: {model_path}")
            
            # Load using ESM's official loader by model name
            self.msa_transformer, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
            print("Successfully loaded MSA model using ESM pretrained loader")
            for param in self.msa_transformer.parameters():
                param.requires_grad = False
                
            self.msa_transformer = self.msa_transformer.to(device)
            print(f"Successfully loaded MSA Transformer. Model has {sum(p.numel() for p in self.msa_transformer.parameters())} parameters")
            
        except Exception as e:
            print(f"Error loading MSA Transformer: {e}")
            print(f"Model path: {model_path}")
            raise
        # ===========================================Load the MSA Transformer model===========================================

        self.msa_out_dim = self.msa_transformer.args.embed_dim

        
        
    def forward(self, msa_tokens):
        """
        使用MSA Transformer处理批量MSA token数据
        
        Args:
            msa_tokens: 已经tokenized的MSA数据 [batch_size, num_seqs, seq_len]
            
        Returns:
            msa_embeddings: MSA第一条序列的特征向量，[batch_size, seq_len, output_dim]，没有去除cls/bos, eos, pad
            valid_tokens_mask: Boolean mask for valid tokens [batch_size, seq_len]
        """
        if msa_tokens is None or msa_tokens.size(0) == 0:
            print("MSA tokens are None or empty")
            return None, None
        try:
            # Process through MSA Transformer
            # Ensure tokens are on the same device as the model
            if hasattr(self.msa_transformer, 'embed_tokens') and hasattr(self.msa_transformer.embed_tokens, 'weight'):
                model_device = self.msa_transformer.embed_tokens.weight.device
                msa_tokens = msa_tokens.to(model_device)
            
            # results is a dict, results["representations"][self.msa_transformer.num_layers] is the hidden features of the last layer, its shape is: (batch_size, msa_depth, seq_len, embed_dim)
            #       batch_size: number of MSA in a batch
            #       msa_depth: number of sequences in each MSA (alignment depth)
            #       seq_len: length of each sequence (including special tokens, such as CLS/EOS)
            #       embed_dim: embedding dimension of the model (such as 768)
            results = self.msa_transformer(msa_tokens, repr_layers=[self.msa_transformer.num_layers])
            msa_representations = results["representations"][self.msa_transformer.num_layers]
            
            # Use the representation of the first sequence (query sequence) in each MSA !!!
            seq_repr = msa_representations[:, 0, :, :]  # [batch_size, seq_len, embed_dim]

            valid_tokens_mask = torch.ones_like(msa_tokens[:, 0, :], dtype=torch.bool)
            
            # Filter out special tokens (consistent with EvolutionaryScaleModeling)
            if hasattr(self.alphabet, 'cls_idx'):
                valid_tokens_mask &= (msa_tokens[:, 0, :] != self.alphabet.cls_idx)
            if hasattr(self.alphabet, 'eos_idx'):
                valid_tokens_mask &= (msa_tokens[:, 0, :] != self.alphabet.eos_idx)
            valid_tokens_mask &= (msa_tokens[:, 0, :] != self.alphabet.padding_idx)
            if hasattr(self.alphabet, 'mask_idx'):
                valid_tokens_mask &= (msa_tokens[:, 0, :] != self.alphabet.mask_idx)
            if hasattr(self.alphabet, 'unk_idx'):
                valid_tokens_mask &= (msa_tokens[:, 0, :] != self.alphabet.unk_idx)
            
            # return seq_repr, valid_tokens_mask
            return seq_repr, valid_tokens_mask
            
        except Exception as e:
            print(f"Error processing MSA data: {e}")
            raise
