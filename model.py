import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, num_hidden_layer=100, input_dim=10, hidden_dim=10, output_dim=1, ):
        super(DNN, self).__init__()
        # self.num_hidden_layer = num_hidden_layer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # self.single_hidden_layer = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Sigmoid())

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        # self.hidden_layer = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim)) for i in range(self.num_hidden_layer)])
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, input_info):
        input = input_info['input']
        x = self.input_layer(input)
        x = F.relu(x)
        # for hidden_layer in self.hidden_layer:
        #     x = hidden_layer(x)
        x = self.output_layer(x)
        output = F.sigmoid(x)

        return output


class Net(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1, num_hidden_layer=100, activation_func=nn.Sigmoid(), hidden_activation_func=nn.Sigmoid(), is_residual=False):
        super(Net,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layer = num_hidden_layer
        self.activation_func = activation_func
        self.hidden_activation_func = hidden_activation_func
        self.is_residual = is_residual
        if self.is_residual:
            self.layer_norm = nn.LayerNorm(self.input_dim, eps=1e-5, elementwise_affine=True)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        if self.is_residual:
            self.hidden_layer = nn.ModuleList([
                nn.Sequential(
                    self.layer_norm,
                    nn.Linear(hidden_dim, hidden_dim),
                    hidden_activation_func) for i in range(self.num_hidden_layer)
            ])
        else:
            self.hidden_layer = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    hidden_activation_func) for i in range(self.num_hidden_layer)
            ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, input):
        x = input['input']
        # 残差链接
        if self.is_residual:
            out = self.input_layer(x)
            for hidden_layer in self.hidden_layer:
                out = hidden_layer(out + x)
            out = self.output_layer(self.layer_norm(out + x))
            out = self.activation_func(out)
        else:
            out = self.input_layer(x)
            for hidden_layer in self.hidden_layer:
                out = hidden_layer(out)
            out = self.output_layer(out)
            out = self.activation_func(out)

        return out

