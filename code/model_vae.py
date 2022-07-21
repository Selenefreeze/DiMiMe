import torch
from torch import nn
from torch.nn import Parameter, Linear
from torch_geometric.utils import softmax


# Encoder

class Encoder(torch.nn.Module):
    def __init__(self, B_num, A_num,
                 B_feature_num, A_feature_num):
        super(Encoder, self).__init__()
        self.B_num = B_num
        self.A_num = A_num
       
        self.l_B1 = nn.Linear(B_feature_num, 256)
        self.l_A1 = nn.Linear(A_feature_num, 256)
        
        self.l_B21 = nn.Linear(256, 64)
        self.l_B22 = nn.Linear(256, 64)
        
        self.l_A21 = nn.Linear(256, 64)
        self.l_A22 = nn.Linear(256, 64)
        
        self.act = nn.ReLU()
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, B_feature, A_feature):

        B_feature = self.act(self.l_B1(B_feature))
        A_feature = self.act(self.l_A1(A_feature))
        
        z_B = self.reparameterize(self.l_B21(B_feature),self.l_B22(B_feature))
        z_A = self.reparameterize(self.l_A21(A_feature),self.l_A22(A_feature))
        
        return z_B, z_A
    
    
# Decoder

class Decoder(torch.nn.Module):
    """
    MLP decoder
    return drug-disease pair predictions
    """

    def __init__(self, input_dim):
        super(Decoder, self).__init__()
        self.mlp_B1 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(64, 128),
                                   nn.ReLU())
        self.mlp_A1 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(64, 128),
                                   nn.ReLU())
        
        self.mlp_2 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(256, 128),
                                   nn.ReLU())
        
        self.mlp_3 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(128, 1),
                                   nn.Sigmoid())

    def forward(self, B_feature, A_feature):
        B_feature = self.mlp_B1(B_feature)
        A_feature = self.mlp_A1(A_feature)
        
        pair_feature = torch.cat([B_feature, A_feature], dim=1)
        
        embedding_1 = self.mlp_2(pair_feature)
        outputs = self.mlp_3(embedding_1)
        return outputs


# VAE

class VAE(torch.nn.Module):
    def __init__(self, B_num, A_num, B_feature_num, A_feature_num):
        super(VAE, self).__init__()
        self.encoder = Encoder(B_num, A_num, B_feature_num, A_feature_num)
        self.decoder = Decoder()

    def forward(self, B_num, A_num, B_feature_num, A_feature_num, pair):
        B_feature, A_feature = self.encoder(B_num, A_num, B_feature_num, A_feature_num)
        row, col = pair
        prediction = self.decoder(B_feature[row, :], A_feature[col, :]).flatten()
        return prediction