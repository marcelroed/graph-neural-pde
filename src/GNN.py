import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function


# Define the GNN model.
class GNN(BaseGNN):
    """
    Basic GNN model.

    Forward is comprised of three steps:
        1. forward_XN
        2. GNN_postXN
        3. m2

    forward_XN
        1. Encodes each node based on their feature vectors
        2. Sets the initial state of the ODE solver
            Note that it doesn't seem like the initial state requires gradients
            JAX should probably be able to figure this out at compile time, though.
        3. Sets the attributes of the ODE solver
            i. Construct W and Omega based on the desired "style" (different ways of being symmetric or diagonal or other things)
        4. Solves the ODE using odeblock

    encode
        1. Applies a linear layer to transform the input into the hidden dimension
        (2.) Optionally applies dropout, several layers with relus between them, concatenates with labels, batch norm, augments with zero vector

    GNN_postXN
        1. Deal with potential augmentation
        2. Activation
        3. fc from bottleneck
    """
    def __init__(self, opt, dataset, device=torch.device('cpu')):
        super(GNN, self).__init__(opt, dataset, device)
        self.f = set_function(opt)
        block = set_block(opt)
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
        self.odeblock.odefunc.GNN_postXN = self.GNN_postXN
        self.odeblock.odefunc.GNN_m2 = self.m2

    def encoder(self, x, pos_encoding=None):
        # Encode each node based on its feature.
        if self.opt['use_labels']:
            y = x[:, -self.num_classes:]
            x = x[:, :-self.num_classes]

        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)

        if self.opt['use_mlp']:
            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
            x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

        if self.opt['use_labels']:
            x = torch.cat([x, y], dim=-1)

        if self.opt['batch_norm']:
            x = self.bn_in(x)

        if self.opt['augment']:
            c_aux = torch.zeros(x.shape).to(self.device)
            x = torch.cat([x, c_aux], dim=1)

        return x

    def set_attributes(self, x):
        self.odeblock.odefunc.W = self.odeblock.odefunc.set_W()
        self.odeblock.odefunc.Omega = self.odeblock.odefunc.set_Omega()

    def forward_XN(self, x, pos_encoding=None):
        ###forward XN
        x = self.encoder(x, pos_encoding=None)
        self.odeblock.set_x0(x)
        self.set_attributes(x)

        if self.training and self.odeblock.nreg > 0:
            z, self.reg_states = self.odeblock(x)
        else:
            z = self.odeblock(x)
        return z

    def GNN_postXN(self, z):
        if self.opt['augment']:
            z = torch.split(z, z.shape[1] // 2, dim=1)[0]
        # Activation.
        if not self.opt['XN_activation']:
            z = F.relu(z)
        # fc from bottleneck
        if self.opt['fc_out']:
            z = self.fc(z)
            z = F.relu(z)
        # Dropout.
        z = F.dropout(z, self.opt['dropout'], training=self.training)
        return z

    def forward(self, x, pos_encoding=None):
        z = self.forward_XN(x)
        z = self.GNN_postXN(z)
        # Decode each node embedding to get node label.
        z = self.m2(z)
        return z
