import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from torchdrug import layers, data
from torchdrug.models.gearnet import GeometryAwareRelationalGraphNeuralNetwork as GearNet

# Define a type alias for clarity, representing a single or a batch of proteins
ProteinGraphType = Union[data.Protein, data.PackedGraph]


class StructureEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 512, 512, 512, 512, 512],
        num_relation: int = 7,
        edge_input_dim: Optional[int] = None,
        num_angle_bin: Optional[int] = None,
        batch_norm: bool = True,
        short_cut: bool = True,
        readout: str = "sum",
    ):
        """
        Initialize the StructureEncoder. The default parameters are set to match the
        specified GearNet configuration.

        Args:
            input_dim (int): The dimension of the input node features (e.g., ESM embedding dimension).
            hidden_dims (List[int]): A list of hidden layer dimensions for the GearNet model. hidden_dims[-1] is the output dimension.
            num_relation (int): The number of edge relation types for the graph.
            edge_input_dim (Optional[int]): Dimension of edge features, if provided. Defaults to None. This is REQUIRED if num_angle_bin is set.
            num_angle_bin (Optional[int]): Number of bins for discretizing angles to enable the spatial line graph.
                                           If set to an integer, the line graph message passing is enabled.
                                           Defaults to None (disabled).          
            batch_norm (bool): Whether to use batch normalization in GearNet.
            short_cut (bool): Whether to use shortcut connections in GearNet.
            readout (str): The graph readout method, typically "sum" or "mean".
        """
        super(StructureEncoder, self).__init__()

        # Assert that edge_input_dim is provided when enabling the line graph.
        if num_angle_bin is not None:
            assert edge_input_dim is not None, \
                "edge_input_dim must be specified when num_angle_bin is enabled."

        # Initialize the GearNet model.
        self.structure_model = GearNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_relation=num_relation,
            edge_input_dim=edge_input_dim,
            num_angle_bin=num_angle_bin,
            batch_norm=batch_norm,
            short_cut=short_cut,
            readout=readout,
        )



    def forward(
        self,
        protein_graph: ProteinGraphType,
        node_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the StructureEncoder.

        Args:
            protein_graph (ProteinGraphType): A `torchdrug.data.Protein` or `torch.data.PackedGraph` object.
            node_features (torch.Tensor): A tensor of pre-computed node features.
                                          Shape: (total_num_residues_in_batch, input_dim).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "node_feature": Structure-aware features for each residue.
                                  Shape: (total_num_residues_in_batch, hidden_dims[-1]).
                - "graph_feature": A feature vector for each graph in the batch.
                                   Shape: (batch_size, hidden_dims[-1]).
        """
        graph = protein_graph
        structure_output = self.structure_model(graph, node_features)
        structure_node_feature = structure_output.get("node_feature", structure_output.get("residue_feature"))
        structure_graph_feature = structure_output["graph_feature"]

        return {
            "node_feature": structure_node_feature,
            "graph_feature": structure_graph_feature,
        }


        