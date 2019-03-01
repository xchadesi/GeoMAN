import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import init
from torch import nn

def Linear(args, output_size, bias, bias_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias(default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    """
    
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.data.size(1) for a in args]
    for shape in shapes:
        total_arg_size += shape
    
    # Now the computation.
    weights = nn.Parameter(torch.FloatTensor(total_arg_size, output_size))
    init.xavier_uniform(weights)
    #weights = Variable(torch.zeros(total_arg_size, output_size))
    if len(args) == 1:
        res = torch.matmul(args[0], weights)
    else:
        res = torch.matmul(torch.cat(args, 1), weights)
    if not bias:
        return res
    
    if bias_initializer is None:
        biases = Variable(torch.zeros(output_size))
        
    return torch.add(res, biases)

def basic_hyperparams():
    return {
        # model parameters
        'learning_rate':1e-3,
        'lambda_l2_reg':1e-3,
        'gc_rate':2.5,  # to avoid gradient exploding
        'dropout_rate':0.3,
        'n_stacked_layers':2,
        's_attn_flag':2,
        'ext_flag':False,

        # encoder parameter
        'n_sensors':307,
        'n_input_encoder':3,
        'n_steps_encoder':5,  # time steps
        'n_hidden_encoder':64,  # size of hidden units

        # decoder parameter
        'n_input_decoder':1,
        'n_external_input':83,
        'n_steps_decoder':5,
        'n_hidden_decoder':64,
        'n_output_decoder':1  # size of the decoder output
    }

def load_data(input_path, mode, n_steps_encoder, n_steps_decoder):
    """ load training/validation data 
    Args:
        input_path:
        mode: "train" or "test"
        n_steps_encoder: length of encoder, i.e., how many historical time steps we use for predictions
        n_steps_decoder: length of decoder, i.e., how many future time steps we predict
    Return:
        a list
    """
    mode_local_inp = np.load(
        input_path + "GeoMAN-{}-{}-{}-local_inputs.npy".format(n_steps_encoder, n_steps_decoder, mode))
    global_attn_index = np.load(
        input_path + "GeoMAN-{}-{}-{}-global_attn_state_indics.npy".format(n_steps_encoder, n_steps_decoder, mode))
    global_inp_index = np.load(
        input_path + "GeoMAN-{}-{}-{}-global_input_indics.npy".format(n_steps_encoder, n_steps_decoder, mode))
    mode_ext_inp = np.load(
        input_path + "GeoMAN-{}-{}-{}-external_inputs.npy".format(12, 6, mode))
    mode_labels = np.load(
        input_path + "GeoMAN-{}-{}-{}-decoder_gts.npy".format(n_steps_encoder, n_steps_decoder, mode))
    return [mode_local_inp, global_inp_index, global_attn_index, mode_ext_inp, mode_labels]

def shuffle_data(training_data):
    """ shuffle data"""
    shuffle_index = np.random.permutation(training_data[0].shape[0])
    new_training_data = []
    for inp in training_data:
        new_training_data.append(inp[shuffle_index])
    return new_training_data

def load_global_inputs(input_path, n_steps_encoder, n_steps_decoder):
    """ load global inputs"""
    global_inputs = np.load(
        input_path + "GeoMAN-{}-{}-global_inputs.npy".format(n_steps_encoder, n_steps_decoder))
    global_attn_states = np.load(
        input_path + "GeoMAN-{}-{}-global_attn_state.npy".format(n_steps_encoder, n_steps_decoder))
    return global_inputs, global_attn_states

def get_batch_feed_dict(k, batch_size, training_data, global_inputs, global_attn_states):
    """ get feed_dict of each batch in a training epoch"""
    train_local_inp = training_data[0]
    train_global_inp = training_data[1]
    train_global_attn_ind = training_data[2]
    train_ext_inp = training_data[3]
    train_labels = training_data[4]
    n_steps_encoder = train_local_inp.shape[1]

    batch_local_inp = train_local_inp[k:k + batch_size]
    batch_ext_inp = train_ext_inp[k:k + batch_size]
    batch_labels = train_labels[k:k + batch_size]
    batch_labels = np.expand_dims(batch_labels, axis=2)
    batch_global_inp = train_global_inp[k:k + batch_size]
    batch_global_attn = train_global_attn_ind[k:k + batch_size]
    tmp = []
    for j in batch_global_inp:
        tmp.append(
            global_inputs[j: j + n_steps_encoder, :])
    tmp = np.array(tmp)
    feed_dict = (torch.from_numpy(batch_local_inp), torch.from_numpy(tmp), torch.from_numpy(batch_ext_inp), torch.from_numpy(np.swapaxes(batch_local_inp, 1, 2)), torch.from_numpy(global_attn_states[batch_global_attn]),  torch.from_numpy(batch_labels))
    
    return feed_dict

def get_valid_batch_feed_dict(k, valid_indexes, valid_data, global_inputs, global_attn_states):
    """ get feed_dict of each batch in the validation set"""
    valid_local_inp = valid_data[0]
    valid_global_inp = valid_data[1]
    valid_global_attn_ind = valid_data[2]
    valid_ext_inp = valid_data[3]
    valid_labels = valid_data[4]
    n_steps_encoder = valid_local_inp.shape[1]

    batch_local_inp = valid_local_inp[valid_indexes[k]:valid_indexes[k + 1]]
    batch_ext_inp = valid_ext_inp[valid_indexes[k]:valid_indexes[k + 1]]
    batch_labels = valid_labels[valid_indexes[k]:valid_indexes[k + 1]]
    batch_labels = np.expand_dims(batch_labels, axis=2)
    batch_global_inp = valid_global_inp[valid_indexes[k]:valid_indexes[k + 1]]
    batch_global_attn = valid_global_attn_ind[valid_indexes[k]:valid_indexes[k + 1]]
    
    tp = []
    for j in batch_global_inp:
        tp.append(
            global_inputs[j: j + n_steps_encoder, :])
    tp = np.array(tp)
    
    
    feed_dict =(torch.from_numpy(batch_local_inp),torch.from_numpy(tp),torch.from_numpy(batch_ext_inp),torch.from_numpy(np.swapaxes(batch_local_inp, 1, 2)),torch.from_numpy(global_attn_states[batch_global_attn]),  torch.from_numpy(batch_labels))
    
    return feed_dict