import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self, device_num=30):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 2, out_channels = 128, kernel_size=19, stride=1) # kernel = 3
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 32, kernel_size=15, stride=1) # kernel=3
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 16, kernel_size=11, stride=1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features = 1872, out_features = 128) # 1872
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(in_features = 128, out_features = 64) # out_features = 64 when device_num=30
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(in_features = 64, out_features = device_num) # in_features = 64 when device_num=30
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.train_flag = False
        self.num_segment = 17
        
    def forward(self, x):
        
        if self.train_flag:
            x = x[:,0:1024*self.num_segment,:]
            out = x.permute(0,2,1).reshape(x.size(0),2,self.num_segment,-1).permute(0,2,1,3).reshape(x.size(0)*self.num_segment,2,-1)
        else:
            out = x
        out = F.celu(self.conv1(out))
        out = nn.MaxPool1d(2)(out)
        out = F.celu(self.conv2(out))
        out = nn.MaxPool1d(2)(out)
        out = F.celu(self.conv3(out))
        out = nn.MaxPool1d(2)(out)
        out = out.reshape(-1, 1872)

        out = F.celu(self.fc1(out))
        out = self.dropout(out)
        out = F.celu(self.fc2(out))
        out = self.dropout(out)
        out = F.log_softmax(self.fc3(out), dim=1)
        
        return out
    
def photonic_act(x):
    denom = torch.mul(x,x) + torch.mul(0.3+0.25*x, 0.3+0.25*x) # torch.div(0.3, x) + 0.25
    out = torch.div(torch.mul(x,x), denom)
#     denom = torch.mul(denom, denom) + 1.0
    return out

class PRNN(nn.Module):
    """Continuous-time RNN.

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=5, **kwargs):
        super(PRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 10 # nn.Parameter(torch.linspace(10,12,hidden_size), requires_grad=False)
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        self.input2h = nn.Linear(input_size, hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=True)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        pre_activation = self.input2h(input) + self.h2h(photonic_act(hidden))
        h_new = hidden * self.oneminusalpha +  self.alpha * (pre_activation)
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(photonic_act(hidden))

        output = torch.stack(output, dim=0)
        return output, hidden
    
class PRNNNet(nn.Module):
    """Photonic Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self, input_size=64, hidden_size=16, **kwargs):
        super(PRNNNet, self).__init__()

        # Continuous time RNN
        self.rnn = PRNN(input_size, hidden_size, **kwargs)
        self.conv0 = nn.Conv1d(in_channels = hidden_size, out_channels = 16, kernel_size=5, stride=1)#.to(device)
        torch.nn.init.xavier_uniform_(self.conv0.weight)#.to(device) # v0 (but no maxpool) v1 out_channels= 128, v2 64, v3 16, v4 4 and not using conv1
        self.conv1 = nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size=3, stride=1)#.to(device)
        torch.nn.init.xavier_uniform_(self.conv1.weight)#.to(device)
        self.fc2 = nn.Linear(in_features= 96, out_features=30)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.train_flag = False
        self.num_segment = 17
        
    def forward(self, x):
        
        if self.train_flag:
            x = x[:,0:1024*self.num_segment,:]
            x = x.permute(0,2,1).reshape(x.size(0),2,self.num_segment,-1).permute(0,2,1,3).reshape(x.size(0),self.num_segment,-1)
            x = x.reshape(x.size(0), self.num_segment, 2, 32, 32).permute(0,1,2,4,3).reshape(x.size(0)*self.num_segment, 64, 32).permute(0,2,1)
        else:
            x = x.reshape(x.size(0), 2, 32, 32).permute(0,1,3,2).reshape(x.size(0), 64, 32).permute(0,2,1)
            
        out = x * 625 #/ torch.std(x, dim=2).unsqueeze(2) # * 625
        out = out.permute(1,0,2)
        
        rnn_activity, _ = self.rnn(out) # bs, seq_len, hidden_dim (100, 17, hd)
        out = rnn_activity.permute(1,2,0)
        
        out = self.conv0(out)
        out = F.celu(out)
        out = nn.MaxPool1d(2)(out)
        out = self.conv1(out)
        out = F.celu(out)
        out = nn.MaxPool1d(2)(out)
                
        out = out.reshape(out.size(0),-1)#out.permute(0,2,1)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)

        return out