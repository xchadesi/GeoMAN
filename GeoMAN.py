import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
from utils import Linear
from torch.nn import init

def input_transform(x):
    local_inputs,global_inputs,external_inputs,local_attn_states,global_attn_states,labels = x
    batch_size = labels.data.size(0)
    n_steps_decoder = labels.data.size(1)
    n_output_decoder = labels.data.size(2)
    n_sensors = global_inputs.data.size(2)
    n_steps_encoder = local_inputs.data.size(1)
    n_input_encoder = local_inputs.data.size(2)
    n_external_input = external_inputs.data.size(2)

    # a tuple composed of the local and global attention states
    #print(local_attn_states.size(), global_attn_states.size())
    encoder_attention_states = (local_attn_states, global_attn_states)

    # transform the inputs from local and global view into encoder_inputs
    _local_inputs = local_inputs.permute(1, 0, 2)
    _local_inputs = _local_inputs.contiguous().view(-1, n_input_encoder)
    _local_inputs = torch.split(_local_inputs, batch_size, 0)
    _global_inputs = global_inputs.permute(1, 0, 2)
    _global_inputs = _global_inputs.contiguous().view(-1, n_sensors)
    _global_inputs = torch.split(_global_inputs, batch_size, 0)
    encoder_inputs = (_local_inputs, _global_inputs)

    # transform the variables into lists as the input of different function
    _labels = labels.permute(1, 0, 2)
    _labels = _labels.contiguous().view(-1, n_output_decoder)
    _labels = torch.split(_labels, batch_size, 0)
    #print(_labels[0].size())
    _external_inputs = external_inputs.permute(1, 0, 2)
    _external_inputs = _external_inputs.contiguous().view(-1, n_external_input)
    _external_inputs = torch.split(_external_inputs, batch_size, 0)
    #print(_external_inputs[0].size())
    # not useful when the loop function is employed
    decoder_inputs = [torch.zeros_like(_labels[0])] + list(_labels[:-1])
    
    return encoder_attention_states, encoder_inputs, _labels, _external_inputs, decoder_inputs

class GeoMAN(nn.Module):
    def __init__(self, hps):
        super(GeoMAN, self).__init__()
        self.hps = hps
        self.w_out = nn.Parameter(torch.FloatTensor(hps['n_hidden_decoder'], 
                                                    hps['n_output_decoder']))
        self.b_out = nn.Parameter(torch.FloatTensor(hps['n_output_decoder']))
    
        self.encoder_cell = nn.LSTMCell(hps['n_input_encoder']+hps['n_sensors'], hps['n_hidden_encoder'], bias=True)
        self.decoder_cell = nn.LSTMCell(hps['n_input_decoder'], hps['n_hidden_decoder'], bias=True)
        init.xavier_uniform(self.w_out)
        init.normal(self.b_out)                          

    def spatial_attention(self, encoder_inputs, attention_states, cell, s_attn_flag=2, output_size=64):
        
        local_inputs = encoder_inputs[0]
        #print(local_inputs[0].size())
        global_inputs = encoder_inputs[1]
        local_attention_states = attention_states[0]
        global_attention_states = attention_states[1]
        #local_inputs is a tuple
        batch_size = local_inputs[0].size(0)
        #print(batch_size)
        
        # decide whether to use local/global attention
        # s_attn_flag: 0: only local. 1: only global. 2: local + global
        local_flag = True
        global_flag = True
        if s_attn_flag == 0:
            global_flag = False
        elif s_attn_flag == 1:
            local_flag = False
        
        if local_flag:
            local_attn_length = local_attention_states.data.size(1) # n_input_encoder
            local_attn_size = local_attention_states.data.size(2) # n_steps_encoder
            # A trick: to calculate U_l * x^{i,k} by a 1-by-1 convolution
            local_hidden = local_attention_states.contiguous().view(-1, local_attn_size, local_attn_length, 1)
            # Size of query vectors for attention.
            local_attention_vec_size = local_attn_size
            local_u = nn.Conv2d(local_attn_size, local_attention_vec_size, (1,1), (1, 1))
            #print(local_hidden.size())
            local_hidden_features = local_u(local_hidden.float())
            
            local_v = nn.Parameter(torch.FloatTensor(local_attention_vec_size))                   
            #local_v = Variable(torch.zeros(local_attention_vec_size)) # v_l

            #local_attn = Variable(torch.zeros(batch_size, local_attn_length))
            local_attn = nn.Parameter(torch.FloatTensor(batch_size, local_attn_length))  
            init.normal(local_v)
            init.xavier_uniform(local_attn)  
            
            def local_attention(query):
                # linear map
                y = Linear(query, local_attention_vec_size, True)
                y = y.view(-1, 1, 1, local_attention_vec_size)
                # Attention mask is a softmax of v_l^{\top} * tanh(...)
                #print((local_v * torch.tanh(local_hidden_features + y)).size())
                s = torch.sum(local_v * torch.tanh(local_hidden_features + y), dim=[1, 3])
                # Now calculate the attention-weighted vector, i.e., alpha in eq.[2]
                a = tf.softmax(s)
                return a
        
        if global_flag:
            global_attn_length = global_attention_states.data.size(1) # n_input_encoder
            global_n_input = global_attention_states.data.size(2)
            global_attn_size = global_attention_states.data.size(3) # n_steps_encoder

            # A trick: to calculate U_l * x^{i,k} by a 1-by-1 convolution
            global_hidden = global_attention_states.contiguous().view(-1, global_attn_size, global_attn_length, global_n_input)
            # Size of query vectors for attention.
            global_attention_vec_size = global_attn_size
            global_k = nn.Conv2d(global_attn_size, global_attention_vec_size, (1,global_n_input), (1, 1))
            global_hidden_features = global_k(global_hidden.float())

            #global_v = Variable(torch.zeros(global_attention_vec_size)) # v_l
            global_v = nn.Parameter(torch.FloatTensor(global_attention_vec_size)) 

            #global_attn = Variable(torch.zeros(batch_size, global_attn_length))
            global_attn = nn.Parameter(torch.FloatTensor(batch_size, global_attn_length)) 
            init.normal(global_v)                        
            init.xavier_uniform(global_attn)
                                       
            def global_attention(query):
                # linear map
                y = Linear(query, global_attention_vec_size, True)
                y = y.view(-1, 1, 1, global_attention_vec_size)
                # Attention mask is a softmax of v_g^{\top} * tanh(...)
                s = torch.sum(global_v * torch.tanh(global_hidden_features + y), dim=[1, 3])
                a = tf.softmax(s)
                
                return a
        
        outputs = []
        attn_weights = []
        i = 0
        # i is the index of the which time step
        # local_inp is numpy.array and the shape of local_inp is (batch_size, n_feature)
        for local_inp, global_inp in zip(local_inputs, global_inputs):
            if local_flag and global_flag:
                # multiply attention weights with the original input
                #print(global_attn.size(), global_inp.size())
                #print(local_attn.size(), local_inp.size())
                local_x = local_attn * local_inp.float()
                
                global_x = global_attn * global_inp.float()
                # Run the BasicLSTM with the newly input
                xx = torch.cat([local_x, global_x], 1)
                #print(xx.size())
                cell_output, state = cell(xx)
                # Run the attention mechanism.
                #print(state.size())
                local_attn = local_attention([state])
                #print(local_attn.size())
                global_attn = global_attention([state])
                attn_weights.append((local_attn, global_attn))
            elif local_flag:
                local_x = local_attn * local_inp
                cell_output, state = cell(local_x)
                local_attn = local_attention([state])
                attn_weights.append(local_attn)
            elif global_flag:
                global_x = global_attn * global_inp
                cell_output, state = cell(global_x)
                global_attn = global_attention([state])
                attn_weights.append(global_attn)
            # Attention output projection
            output = cell_output
            outputs.append(output)
            i += 1
            
        return outputs, state, attn_weights

    def temporal_attention(self, decoder_inputs, external_inputs, encoder_state, attention_states, 
                           cell, external_flag, output_size=64):
            # Needed for reshaping.
            batch_size = decoder_inputs[0].data.size(0)
            attn_length = attention_states.data.size(1)
            attn_size = attention_states.data.size(2)
            
            # A trick: to calculate W_d * h_o by a 1-by-1 convolution
            # See at eq.[6] in the paper
            hidden = attention_states.view(-1, attn_size, attn_length, 1) # need to reshape before
            # Size of query vectors for attention.
            attention_vec_size = attn_size
            w_conv = nn.Conv2d(attn_size, attention_vec_size, (1,1), (1,1))
            hidden_features = w_conv(hidden) 
            #v = Variable(torch.zeros(attention_vec_size)) # v_l
            v = nn.Parameter(torch.FloatTensor(attention_vec_size)) 
            init.normal(v)       
                             
            def attention(query):
                # linear map
                y = Linear(query, attention_vec_size, True)
                y = y.view(-1, 1, 1, attention_vec_size)
                # Attention mask is a softmax of v_d^{\top} * tanh(...).
                s = torch.sum(v * torch.tanh(hidden_features + y), dim=[1, 3])
                # Now calculate the attention-weighted vector, i.e., gamma in eq.[7]
                a = tf.softmax(s)
                # eq. [8]
                #print(hidden.size())
                #print((a.view(-1, 1, attn_length, 1)).size())
                d = torch.sum(a.view(-1, 1, attn_length, 1)* hidden, dim=[2, 3])
                    
                return d.view(-1, attn_size)
            
            #attn = Variable(torch.zeros(batch_size, attn_size))
            attn = nn.Parameter(torch.FloatTensor(batch_size, attn_size)) 
            init.xavier_uniform(attn)    
                             
            i = 0
            outputs = []
            prev = None
            
            for (inp, ext_inp) in zip(decoder_inputs, external_inputs):
                # Merge input and previous attentions into one vector of the right size.
                
                input_size = inp.data.size(1)
                #print(i, input_size)
                #input_size是指向量维度
                # we map the concatenation to shape [batch_size, input_size]
                if external_flag:
                    #print(inp.data.size(1),ext_inp.data.size(1),attn.data.size(1))
                    x = Linear([inp.float()] + [ext_inp.float()] + [attn.float()], input_size, True)
                else:
                    x = Linear([inp.float()] + [attn.float()], input_size, True)
                # Run the RNN.
                #print(x.size())
                cell_output, state = cell(x)
                # Run the attention mechanism.
                #print(state.size())
                attn = attention([state])
                
                # Attention output projection
                #print(cell_output.size(), attn.size())
                output = Linear([cell_output] + [attn], output_size, True)
                outputs.append(output)
                i += 1
            return outputs, state
    
    def forward(self, x):
        encoder_attention_states, encoder_inputs, _labels, _external_inputs, decoder_inputs \
            = input_transform(x)
        
        encoder_outputs, encoder_state, attn_weights = self.spatial_attention(encoder_inputs,
                                                                              encoder_attention_states,
                                                                              self.encoder_cell,
                                                                              self.hps['s_attn_flag'])

        # Calculate a concatenation of encoder outputs to put attention on.
        top_states = [e.view(-1, 1, 64) for e in encoder_outputs]
        attention_states = torch.cat(top_states, 1)

        # the implement of decoder
        decoder_outputs, states = self.temporal_attention(decoder_inputs,
                                                          _external_inputs,
                                                          encoder_state,
                                                          attention_states,
                                                          self.decoder_cell,
                                                          self.hps['ext_flag'])
        
        preds = [torch.matmul(i, self.w_out) + self.b_out for i in decoder_outputs]
        
        return preds, _labels