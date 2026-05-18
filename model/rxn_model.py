from collections import namedtuple
import math
import torch
import torch.nn as nn
from dgllife.model import MPNNGNN
import dgl
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from common.utils import GELU, GlobalMultiHeadAttention



def pack_atom_feats(bg, atom_feats):
    bg.ndata['h'] = atom_feats
    gs = dgl.unbatch(bg)
    edit_feats = [g.ndata['h'] for g in gs]
    masks = [
        torch.ones(g.num_nodes(), dtype=torch.uint8, device=bg.device)
        for g in gs
    ]
    padded_feats = pad_sequence(edit_feats, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)
    return padded_feats, masks



class ReactionMGMTurnNet(nn.Module):
    def __init__(self,
                 node_in_feats,
                 node_out_feats,
                 edge_in_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers,
                 dropout,
                 num_atom_types,
                 cross_attn_h_rate,
                 output_attention=False,
                 is_pretrain=True) -> None:
        super(ReactionMGMTurnNet, self).__init__()
        self.output_attention = output_attention
        self.activation = GELU()

        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                            node_out_feats=node_out_feats,
                            edge_in_feats=edge_in_feats,
                            edge_hidden_feats=edge_hidden_feats,
                            num_step_message_passing=num_step_message_passing)

        self.atom_attn_net = GlobalMultiHeadAttention(
            d_model=node_out_feats,
            heads=attention_heads,
            n_layers=attention_layers,
            positional_number=8,
            cross_attn_h_rate=cross_attn_h_rate,
            dropout=dropout)
        if is_pretrain:
            self.logic_atom_net = nn.Sequential(
                nn.Linear(node_out_feats, node_out_feats * 2), GELU(),
                nn.Dropout(dropout), nn.Linear(node_out_feats * 2, num_atom_types))
        self.node_out_dim = node_out_feats

    def embedded_net(
        self,
        rts_bg,
        rts_node_feats,
        rts_edge_feats,
        pds_bg,
        pds_node_feats,
        pds_edge_feats,
    ):

        rts_atom_feats = self.mpnn(rts_bg, rts_node_feats, rts_edge_feats)
        rts_atom_feats, rts_mask = pack_atom_feats(rts_bg, rts_atom_feats)

        pds_atom_feats = self.mpnn(pds_bg, pds_node_feats, pds_edge_feats)
        pds_atom_feats, pds_mask = pack_atom_feats(pds_bg, pds_atom_feats)

        return rts_atom_feats, rts_mask, pds_atom_feats, pds_mask

    def forward(self, 
                rts_bg, 
                rts_adms, 
                rts_node_feats, 
                rts_edge_feats, 
                pds_bg,
                pds_adms, 
                pds_node_feats, 
                pds_edge_feats,
                rxn_smiles,
                return_rts=False):
        rts_atom_feats, rts_mask, pds_atom_feats, pds_mask = self.embedded_net(
            rts_bg,
            rts_node_feats,
            rts_edge_feats,
            pds_bg,
            pds_node_feats,
            pds_edge_feats,
        )
        rts_atom_feats_h_cross_update, rts_self_att_scores, rts_self_attn_mask, rts_atom_cross_attention, rts_atom_cross_attention_mask = self.atom_attn_net(
            src=rts_atom_feats,
            tgt=pds_atom_feats,
            rpm=rts_adms,
            src_mask=rts_mask,
            tgt_mask=pds_mask)
        if return_rts:
            return rts_atom_feats_h_cross_update, rts_mask

        pds_atom_feats_h_cross_update, pds_self_att_scores, pds_self_attn_mask, pds_atom_cross_attention, pds_atom_cross_attention_mask = self.atom_attn_net(
            src=pds_atom_feats,
            tgt=rts_atom_feats,
            rpm=pds_adms,
            src_mask=pds_mask,
            tgt_mask=rts_mask)

        rts_atom_feats_h_cross_update = rts_atom_feats_h_cross_update[
            rts_mask.bool()]
        pds_atom_feats_h_cross_update = pds_atom_feats_h_cross_update[
            pds_mask.bool()]

        atom_feats_h_cross_update = torch.cat(
            [rts_atom_feats_h_cross_update, pds_atom_feats_h_cross_update],
            dim=0)
        atom_logic = self.logic_atom_net(atom_feats_h_cross_update)

        if self.output_attention:
            attentions_tuple = namedtuple('attentions', [
                'rts_atom_cross_attention', 'pds_atom_cross_attention',

                'rts_atom_cross_attention_mask',
                'pds_atom_cross_attention_mask'
            ])
            attentions = attentions_tuple(
                rts_atom_cross_attention=rts_atom_cross_attention,
                pds_atom_cross_attention=pds_atom_cross_attention,
                rts_atom_cross_attention_mask=rts_atom_cross_attention_mask,
                pds_atom_cross_attention_mask=pds_atom_cross_attention_mask)

            return atom_logic, attentions
        else:
            return atom_logic, None

