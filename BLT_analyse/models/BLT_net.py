import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BLT_net_64(nn.Module):
    
    def __init__(self, lateral_connections = True, topdown_connections = True, LT_interaction = 'additive', timesteps = 2, LT_position = 'all',
                 classifier_bias = True, norm_type = 'LN'):
        super(BLT_net_64, self).__init__()
        # Ensure that kernel_size is always odd, otherwise TransposeConv2d computation will fail
        layer_bias = False
        if norm_type == 'None':
            layer_bias = True
        lt_flag_prelast = 1
        if LT_position == 'last':
            lt_flag_prelast = 0 # if 'last' then only GAP gets a lateral connection
        self.retina = nn.Conv2d(3, 32, 3, padding = 'same')
        # no pooling after retina
        self.v1 = BLT_Conv(32, 64, 3, lateral_connections*lt_flag_prelast, topdown_connections*lt_flag_prelast,
                           LT_interaction, pool_input = False, bias = layer_bias)
        if norm_type == 'LN':
            self.v1_ln = nn.LayerNorm([64,64,64])
        elif norm_type == 'None':
            self.v1_ln = nn.Identity()
        self.v2 = BLT_Conv(64, 128, 3, lateral_connections*lt_flag_prelast, topdown_connections*lt_flag_prelast,
                           LT_interaction, bias = layer_bias)
        if norm_type == 'LN':
            self.v2_ln = nn.LayerNorm([128,32,32])
        elif norm_type == 'None':
            self.v2_ln = nn.Identity()
        self.v4 = BLT_Conv(128, 256, 3, lateral_connections*lt_flag_prelast, topdown_connections*lt_flag_prelast,
                           LT_interaction, bias = layer_bias)
        if norm_type == 'LN':
            self.v4_ln = nn.LayerNorm([256,16,16])
        elif norm_type == 'None':
            self.v4_ln = nn.Identity()
        # IT only has lateral connections
        self.it = BLT_Conv(256, 512, 3, lateral_connections*lt_flag_prelast, False, LT_interaction, bias = layer_bias)
        if norm_type == 'LN':
            self.it_ln = nn.LayerNorm([512,8,8])
        elif norm_type == 'None':
            self.it_ln = nn.Identity()
        # gap layer globalavgpools IT and has a self (lateral) connection
        self.gap = BLT_GAP(8, 512, lateral_connections, LT_interaction, bias = layer_bias)
        if norm_type == 'LN':
            self.gap_ln = nn.LayerNorm([512])
        elif norm_type == 'None':
            self.gap_ln = nn.Identity()
        self.readout = nn.Linear(512, 100, bias=classifier_bias)
        self.timesteps = timesteps
    
    def forward(self, inputs):
        activations = [[None for t in range(self.timesteps)] for i in range(5)]
        outputs = [None for l in range(self.timesteps)]
        retina_out = F.relu(self.retina(inputs))
        for t in range(self.timesteps):
            activations[0][t] = F.relu(self.v1_ln(self.v1(retina_out)))
            if t == 0: # does not accept None as input
                activations[1][t] = F.relu(self.v2_ln(self.v2(activations[0][t])))
                activations[2][t] = F.relu(self.v4_ln(self.v4(activations[1][t])))
                activations[3][t] = F.relu(self.it_ln(self.it(activations[2][t])))
                activations[4][t] = F.relu(self.gap_ln(self.gap(activations[3][t]))) 
            else:
                activations[1][t] = F.relu(self.v2_ln(self.v2(activations[0][t],activations[1][t-1],activations[2][t-1])))
                activations[2][t] = F.relu(self.v4_ln(self.v4(activations[1][t],activations[2][t-1],activations[3][t-1])))
                activations[3][t] = F.relu(self.it_ln(self.it(activations[2][t],activations[3][t-1])))
                activations[4][t] = F.relu(self.gap_ln(self.gap(activations[3][t],activations[4][t-1]))) 
            outputs[t] = torch.log(torch.clamp(F.softmax(self.readout(activations[4][t]),dim=1),1e-10,1.0))
        return outputs, activations[4]
    
class BLT_net_128(nn.Module):
    
    def __init__(self, lateral_connections = True, topdown_connections = True, LT_interaction = 'additive', timesteps = 2, LT_position = 'all',
                 classifier_bias = True, norm_type = 'LN'):
        super(BLT_net_128, self).__init__()
        # Ensure that kernel_size is always odd, otherwise TransposeConv2d computation will fail
        lt_flag_prelast = 1
        if LT_position == 'last':
            lt_flag_prelast = 0 # if 'last' then only GAP gets a lateral connection
        self.retina = nn.Conv2d(3, 32, 5, padding = 'same')
        # no pooling after retina
        self.v1 = BLT_Conv(32, 64, 5, lateral_connections*lt_flag_prelast, topdown_connections*lt_flag_prelast,
                           LT_interaction, False)
        if norm_type == 'LN':
            self.v1_ln = nn.LayerNorm([64,128,128])
        elif norm_type == 'None':
            self.gap_ln = nn.Identity()
        self.v2 = BLT_Conv(64, 128, 5, lateral_connections*lt_flag_prelast, topdown_connections*lt_flag_prelast,
                           LT_interaction)
        if norm_type == 'LN':
            self.v2_ln = nn.LayerNorm([128,64,64])
        elif norm_type == 'None':
            self.gap_ln = nn.Identity()
        self.v4 = BLT_Conv(128, 256, 5, lateral_connections*lt_flag_prelast, topdown_connections*lt_flag_prelast,
                           LT_interaction)
        if norm_type == 'LN':
            self.v4_ln = nn.LayerNorm([256,32,32])
        elif norm_type == 'None':
            self.gap_ln = nn.Identity()
        # IT only has lateral connections
        self.it = BLT_Conv(256, 512, 5, lateral_connections*lt_flag_prelast, False, LT_interaction)
        if norm_type == 'LN':
            self.it_ln = nn.LayerNorm([512,16,16])
        elif norm_type == 'None':
            self.gap_ln = nn.Identity()
        # gap layer globalavgpools IT and has a self (lateral) connection
        self.gap = BLT_GAP(16, 512, lateral_connections, LT_interaction)
        if norm_type == 'LN':
            self.gap_ln = nn.LayerNorm([512])
        elif norm_type == 'None':
            self.gap_ln = nn.Identity()
        self.readout = nn.Linear(512, 100, bias=classifier_bias)
        self.timesteps = timesteps
    
    def forward(self, inputs):
        activations = [[None for t in range(self.timesteps)] for i in range(5)]
        outputs = [None for l in range(self.timesteps)]
        retina_out = F.relu(self.retina(inputs))
        for t in range(self.timesteps):
            activations[0][t] = F.relu(self.v1_ln(self.v1(retina_out)))
            if t == 0: # does not accept None as input
                activations[1][t] = F.relu(self.v2_ln(self.v2(activations[0][t])))
                activations[2][t] = F.relu(self.v4_ln(self.v4(activations[1][t])))
                activations[3][t] = F.relu(self.it_ln(self.it(activations[2][t])))
                activations[4][t] = F.relu(self.gap_ln(self.gap(activations[3][t]))) 
            else:
                activations[1][t] = F.relu(self.v2_ln(self.v2(activations[0][t],activations[1][t-1],activations[2][t-1])))
                activations[2][t] = F.relu(self.v4_ln(self.v4(activations[1][t],activations[2][t-1],activations[3][t-1])))
                activations[3][t] = F.relu(self.it_ln(self.it(activations[2][t],activations[3][t-1])))
                activations[4][t] = F.relu(self.gap_ln(self.gap(activations[3][t],activations[4][t-1]))) 
            outputs[t] = torch.log(torch.clamp(F.softmax(self.readout(activations[4][t]),dim=1),1e-10,1.0))
        return outputs
    
class BLT_Conv(nn.Module):
    # This Conv class takes the input (which can be due to b, l, and/or t) and outputs b, l, t drives for same/other
    # layers.
    
    def __init__(self, in_chan, out_chan, kernel_size, lateral_connection = True, topdown_connection = True,
                 LT_interaction = 'additive', pool_input = True, bias = False):
        super(BLT_Conv, self).__init__()
        self.bottom_up = nn.Conv2d(in_chan, out_chan, kernel_size, bias = bias, padding = 'same')
        self.lateral_connect = lateral_connection
        if lateral_connection:
            self.lateral = nn.Conv2d(out_chan, out_chan, kernel_size, bias = bias, padding = 'same')
        self.topdown_connect = topdown_connection
        if topdown_connection:
            ct_padding = int((kernel_size-1)/2)
            self.top_down = nn.ConvTranspose2d(out_chan*2, out_chan, kernel_size, stride = 2, bias = bias, padding = ct_padding, output_padding = 1)
        self.pool_input = pool_input
        if pool_input:
            self.pool = nn.MaxPool2d(2, 2)
        self.LT_interaction = LT_interaction

    def forward(self, b_input, l_input = None, t_input = None):
        if self.pool_input:
            b_input = self.pool(b_input)
        b_input = self.bottom_up(b_input)
        if self.lateral_connect:
            if l_input is not None:
                l_input = self.lateral(l_input)
        else:
            l_input = None
        if self.topdown_connect:
            if t_input is not None:
                t_input = self.top_down(t_input)
        else:
            t_input = None
            
        if l_input is not None and t_input is not None:
            if self.LT_interaction == 'additive':
                output = b_input + l_input + t_input
            else:
                output = b_input*(1. + l_input + t_input)
        elif l_input is not None:
            if self.LT_interaction == 'additive':
                output = b_input + l_input
            else:
                output = b_input*(1. + l_input)
        elif t_input is not None:
            if self.LT_interaction == 'additive':
                output = b_input + t_input
            else:
                output = b_input*(1. + t_input)
        else:
            output = b_input
        
        return output
    
class BLT_GAP(nn.Module):
    # This GAP class takes the final conv and avgpools and constructs a lateral connection on it
    
    def __init__(self, pool_size, out_chan, lateral_connection = True, LT_interaction = 'additive', bias = False):
        super(BLT_GAP, self).__init__()
        self.lateral_connect = lateral_connection
        if lateral_connection:
            self.lateral = nn.Linear(out_chan, out_chan, bias=bias)
        self.pool = nn.AvgPool2d(pool_size)
        self.LT_interaction = LT_interaction

    def forward(self, b_input, l_input = None):
        b_input = torch.squeeze(self.pool(b_input))
        if self.lateral_connect:
            if l_input is not None:
                l_input = self.lateral(l_input)
        else:
            l_input = None
            
        if l_input is not None:
            if self.LT_interaction == 'additive':
                output = b_input + l_input
            else:
                output = b_input*(1. + l_input)
        else:
            output = b_input 
                
        return output