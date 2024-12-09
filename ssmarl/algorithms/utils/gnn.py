import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import MessagePassing, TransformerConv

class Embedding(MessagePassing):
    def __init__(self,
                 hidden_size: int,
                 edge_dim: int = 0,
                 device: str = 'cpu'):
        super(Embedding, self).__init__(aggr='add')
        self.active_func = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.init_method = nn.init.orthogonal_
        self.entity_embed = nn.Embedding(5, 4)
        self.lin1 = nn.Linear(4 + edge_dim, hidden_size)

        # Initialize the hidden layers
        self.layers = nn.ModuleList()
        for _ in range(2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.active_func)
            self.layers.append(self.layer_norm)

        self.to(device)
        # Apply initialization
        self._initialize_weights()

    def _initialize_weights(self):
        gain = nn.init.calculate_gain('relu' if isinstance(self.active_func, nn.ReLU) else 'tanh')
        self.init_method(self.lin1.weight, gain=gain)
        nn.init.constant_(self.lin1.bias, 0)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.init_method(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, torch.Tensor):
            x: torch.OptPairTensor = (x, x)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        entity_type_j = x_j[:,-1].long()
        entity_embed_j = self.entity_embed(entity_type_j)

        node_feat = torch.cat([entity_embed_j, edge_attr], dim=1)

        # Apply the first linear layer
        x = self.lin1(node_feat)
        x = self.active_func(x)
        x = self.layer_norm(x)

        # Apply the hidden layers
        for layer in self.layers:
            x = layer(x)

        return x

class GNNbase(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, edge_dim, graph_aggr, device='cpu'):
        super(GNNbase, self).__init__()
        # in_channels is the dimension of node_feat
        self.device = device
        self.graph_aggr = graph_aggr

        self.EmbeddingLayer = Embedding(16, edge_dim, device=device)
        # concat is a combination of multi-head attention, concatenation and addition,
        # concatenation will affect the dimensions of the output
        self.GAT1 = TransformerConv(16, 16, heads=num_heads, concat=False, edge_dim=edge_dim)
        self.GAT2 = TransformerConv(16, out_channels, heads=num_heads, concat=False, edge_dim=edge_dim)
        self.activation = nn.ReLU()
        self.to(device)
        self.init_network()

    def init_network(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, nodes_feats, edge_index, edge_attr, agent_id):
        batch_size, num_nodes, _ = nodes_feats.shape

        edge_index_adder = torch.zeros(edge_index.shape,device=self.device)
        i_values = torch.arange(batch_size,device=self.device)
        edge_index_adder[...] = num_nodes * i_values[:, None, None]
        new_edge_index = edge_index + edge_index_adder
        new_edge_index = new_edge_index.permute(1, 0, 2)

        non_nan_mask = ~torch.isnan(nodes_feats)
        nodes_feats = nodes_feats[non_nan_mask].unsqueeze(-1)
        non_nan_mask = ~torch.isnan(new_edge_index)
        edge_index = new_edge_index[non_nan_mask].reshape(2, -1).long()
        non_nan_mask = ~torch.isnan(edge_attr)
        edge_attr = edge_attr[non_nan_mask].reshape(edge_index.shape[1], -1)

        nodes_feats_embeddings = self.EmbeddingLayer(nodes_feats, edge_index, edge_attr)

        A = self.GAT1(nodes_feats_embeddings, edge_index, edge_attr)
        A = self.GAT2(A, edge_index, edge_attr)
        A = self.activation(A)
        
        if self.graph_aggr == 'agent':
            # Agent Aggregation
            A = A.view(batch_size, num_nodes, -1)
            # Ensure agent_id is a long tensor
            agent_id = agent_id.long()
            A = A.gather(1, agent_id.unsqueeze(-1).expand(-1, -1, A.size(-1))).squeeze(1)
        elif self.graph_aggr == 'graph':
            # Graph Aggregation
            A = A.view(batch_size, num_nodes, -1)
            num_agents = torch.sum(torch.eq(nodes_feats, 0))/batch_size
            A = torch.mean(A[:, :num_agents.long()], dim=1)
        return A


if __name__ == "__main__":
    # Test
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    nodes_feats = torch.tensor([[[0.0], [2.0], [3.0], [4.0]],[[0.0], [2.0], [3.0], [4.0]]], device=device)

    edge_index = torch.tensor([[[0, 1, 2, 3], [0, 0, 0, 0]],[[0, 1, 2, torch.nan], [0, 0, 0, torch.nan]]], device=device)

    edge_attr = torch.tensor([[[5.0, 3.0], [6.0, 7.0], [7.0, 2.0], [8.0, 5.0]],[[6.0, 7.0], [7.0, 2.0], [8.0, 5.0], [torch.nan, torch.nan]]], device=device, requires_grad=True)

    G = GNNbase(1, 4, 1, 2, 'global',device)

    rst = G(nodes_feats, edge_index, edge_attr, None)

    print(rst)