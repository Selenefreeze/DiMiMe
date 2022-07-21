import torch
from torch import nn
from torch.nn import Parameter, Linear
from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils import glorot, zeros


# Bipartite Graph Attention

class BipartiteGAT(MessagePassing):

    def __init__(self, in_channels_i, in_channels_j, out_channels, heads=1,
                 attention_concat=False, multi_head_concat=False,
                 negative_slope=0.2, dropout=0.0, bias=True, **kwargs):
        """
        :param in_channels: Size of each input sample.
        :param out_channels: Size of each output sample.
        :param heads: Number of multi-head-attentions.
        :param attention_concat: If set to False, the attentions are only based on one side.
        :param multi_head_concat: If set to False, the multi-head attentions are averaged instead of concatenated.
        :param negative_slope: LeakyReLU angle of the negative.
        :param dropout: Dropout probability of the normalized attention coefficients which exposes each node to a
                        stochastically sampled neighborhood during training.
        :param bias: If set to False, the layer will not learn an additive bias.
        :param kwargs: Additional arguments of `torch_geometric.nn.conv.MessagePassing.
        """
        super(BipartiteGAT, self).__init__(aggr='add', **kwargs)

        self.in_channels_i = in_channels_i
        self.in_channels_j = in_channels_j
        self.out_channels = out_channels
        self.heads = heads
        self.attention_concat = attention_concat
        self.multi_head_concat = multi_head_concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        if attention_concat:
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.att = Parameter(torch.Tensor(1, heads, out_channels))

        self.mlp_i = Linear(in_channels_i, heads * out_channels)
        self.mlp_j = Linear(in_channels_j, heads * out_channels)

        if bias and multi_head_concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not multi_head_concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        x = (self.mlp_i(x[0]), self.mlp_j(x[1]))

        propagate_result = self.propagate(edge_index, size=size, x=x)
        index = 1 if self.flow == "source_to_target" else 0
        final_result = propagate_result + x[index]

        return final_result

    def message(self, edge_index_i, x_i, x_j, size_i):
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if self.attention_concat:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        else:
            alpha = (x_j * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.multi_head_concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



# Encoder

class DiMiMeLayer(torch.nn.Module):
    """
    DiMiMe Encoder Layer
    return embedding vectors of A and B fused by C
    """
    def __init__(self, C_num, B_num, A_num,
                 C_feature_num, B_feature_num, A_feature_num, hidden_dim):
        
        super(DiMiMeLayer, self).__init__()
        self.C_num = C_num
        self.B_num = B_num
        self.A_num = A_num
        self.B_C_gat = BipartiteGAT(B_feature_num, C_feature_num, hidden_dim, heads=1, dropout=0.1,
                                         flow='source_to_target', attention_concat=False, multi_head_concat=False)
        self.A_C_gat = BipartiteGAT(A_feature_num, C_feature_num, hidden_dim, heads=1, dropout=0.1,
                                            flow='source_to_target', attention_concat=False, multi_head_concat=False)

        self.C_C_gat1 = GATConv(2 * hidden_dim, hidden_dim, heads=4, dropout=0.1, concat=False)
        
        self.C_C_gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.1, concat=False)
        
        #self.C_C_gcn = GCNConv(2 * hidden_dim, hidden_dim, dropout=0.1)

        self.C_B_gat = BipartiteGAT(B_feature_num, hidden_dim, hidden_dim, heads=1, dropout=0.1,
                                         flow='target_to_source', attention_concat=False, multi_head_concat=False)
        self.C_A_gat = BipartiteGAT(A_feature_num, hidden_dim, hidden_dim, heads=1, dropout=0.1,
                                            flow='target_to_source', attention_concat=False, multi_head_concat=False)

        self.act = nn.ReLU()
        self.act_tanh = nn.Tanh()

    def forward(self, C_C, B_C, A_C, B_feature, A_feature, C_feature):
        edge_B_C = B_C
        C_feature_from_B = self.act(self.B_C_gat((B_feature, C_feature),
                                                               edge_B_C,
                                                               size=[self.B_num, self.C_num]))

        edge_A_C = A_C
        C_feature_from_A = self.act(self.A_C_gat((A_feature, C_feature),
                                                                     edge_A_C,
                                                                     size=[self.A_num, self.C_num]))

        C_feature = torch.cat([C_feature_from_B, C_feature_from_A], dim=1)
        C_feature = self.act(self.C_C_gat1(C_feature, C_C))
        C_feature = self.act_tanh(self.C_C_gat2(C_feature, C_C))

        B_feature = self.act(self.C_B_gat((B_feature, C_feature),
                                                  edge_B_C,
                                                  size=[self.B_num, self.C_num]))

        A_feature = self.act(self.C_A_gat((A_feature, C_feature),
                                                        edge_A_C,
                                                        size=[self.A_num, self.C_num]))

        return B_feature, A_feature, C_feature
    

    
# Decoder

class Mlp(torch.nn.Module):
    """
    MLP Decoder
    return B-A pair predictions
    """
    def __init__(self, input_dim):
        super(Mlp, self).__init__()
        self.mlp_1 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim * 2), int(input_dim)),
                                   nn.ReLU())
        self.mlp_2 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim), int(input_dim // 2)),
                                   nn.ReLU())
        self.mlp_3 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim // 2), 1),
                                   nn.Sigmoid())

    def forward(self, B_feature, A_feature):
        pair_feature = torch.cat([B_feature, A_feature], dim=1)
        embedding_1 = self.mlp_1(pair_feature)
        embedding_2 = self.mlp_2(embedding_1)
        outputs = self.mlp_3(embedding_2)
        return outputs



# Model

class DiMiMeNet(torch.nn.Module):
    """
    Model structure based on 
    encoder-decoder
    """
    def __init__(self, hidden_dim_1, hidden_dim_2, C_num, B_num, A_num,
                 C_feature_num, B_feature_num, A_feature_num):
        super(DiMiMeNet, self).__init__()
        self.encoder = DiMiMeLayer(C_num, B_num, A_num, C_feature_num, B_feature_num,
                                       A_feature_num, hidden_dim_1)
        self.decoder = Mlp(hidden_dim_1)

    def forward(self, C_C, B_C, A_C, B_feature, A_feature, C_feature, pair):
        B_feature, A_feature, C_feature = self.encoder(C_C, B_C, A_C,
                                                            B_feature, A_feature, C_feature)
        row, col = pair
        prediction = self.decoder(B_feature[row, :], A_feature[col, :]).flatten()
        return prediction