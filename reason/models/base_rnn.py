import torch.nn as nn


class BaseRNN(nn.Module):
    """Base RNN module"""
    
    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, 
                 dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p

        if rnn_cell == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell: %s' % rnn_cell)

        self.input_dropout = nn.Dropout(p=input_dropout_p)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()