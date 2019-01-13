import torch
import torch.nn as nn
from torch.autograd import Variable

from . import create_seq2seq_net, get_vocab
import utils.utils as utils


class Seq2seqParser():
    """Model interface for seq2seq parser"""

    def __init__(self, opt):
        self.opt = opt
        self.vocab = get_vocab(opt)
        if opt.load_checkpoint_path is not None:
            self.load_checkpoint(opt.load_checkpoint_path)
        else:
            print('| creating new network')
            self.net_params = self._get_net_params(self.opt, self.vocab)
            self.seq2seq = create_seq2seq_net(**self.net_params)
        self.variable_lengths = self.net_params['variable_lengths']
        self.end_id = self.net_params['end_id']
        self.gpu_ids = opt.gpu_ids
        self.criterion = nn.NLLLoss()
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            self.seq2seq.cuda(opt.gpu_ids[0])

    def load_checkpoint(self, load_path):
        print('| loading checkpoint from %s' % load_path)
        checkpoint = torch.load(load_path)
        self.net_params = checkpoint['net_params']
        if 'fix_embedding' in vars(self.opt): # To do: change condition input to run mode
            self.net_params['fix_embedding'] = self.opt.fix_embedding
        self.seq2seq = create_seq2seq_net(**self.net_params)
        self.seq2seq.load_state_dict(checkpoint['net_state'])

    def save_checkpoint(self, save_path):
        checkpoint = {
            'net_params': self.net_params,
            'net_state': self.seq2seq.cpu().state_dict()
        }
        torch.save(checkpoint, save_path)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.seq2seq.cuda(self.gpu_ids[0])

    def set_input(self, x, y=None):
        input_lengths, idx_sorted = None, None
        if self.variable_lengths:
            x, y, input_lengths, idx_sorted = self._sort_batch(x, y)
        self.x = self._to_var(x)
        if y is not None:
            self.y = self._to_var(y)
        else:
            self.y = None
        self.input_lengths = input_lengths
        self.idx_sorted = idx_sorted

    def set_reward(self, reward):
        self.reward = reward

    def supervised_forward(self):
        assert self.y is not None, 'Must set y value'
        output_logprob = self.seq2seq(self.x, self.y, self.input_lengths)
        self.loss = self.criterion(output_logprob[:,:-1,:].contiguous().view(-1, output_logprob.size(2)), self.y[:,1:].contiguous().view(-1))
        return self._to_numpy(self.loss).sum()

    def supervised_backward(self):
        assert self.loss is not None, 'Loss not defined, must call supervised_forward first'
        self.loss.backward()

    def reinforce_forward(self):
        self.rl_seq = self.seq2seq.reinforce_forward(self.x, self.input_lengths)
        self.rl_seq = self._restore_order(self.rl_seq.data.cpu())
        self.reward = None # Need to recompute reward from environment each time a new sequence is sampled
        return self.rl_seq

    def reinforce_backward(self, entropy_factor=0.0):
        assert self.reward is not None, 'Must run forward sampling and set reward before REINFORCE'
        self.seq2seq.reinforce_backward(self.reward, entropy_factor)

    def parse(self):
        output_sequence = self.seq2seq.sample_output(self.x, self.input_lengths)
        output_sequence = self._restore_order(output_sequence.data.cpu())
        return output_sequence

    def _get_net_params(self, opt, vocab):
        net_params = {
            'input_vocab_size': len(vocab['question_token_to_idx']),
            'output_vocab_size': len(vocab['program_token_to_idx']),
            'hidden_size': opt.hidden_size,
            'word_vec_dim': opt.word_vec_dim,
            'n_layers': opt.n_layers,
            'bidirectional': opt.bidirectional,
            'variable_lengths': opt.variable_lengths,
            'use_attention': opt.use_attention,
            'encoder_max_len': opt.encoder_max_len,
            'decoder_max_len': opt.decoder_max_len,
            'start_id': opt.start_id,
            'end_id': opt.end_id,
            'word2vec_path': opt.word2vec_path,
            'fix_embedding': opt.fix_embedding,
        }
        return net_params

    def _sort_batch(self, x, y):
        _, lengths = torch.eq(x, self.end_id).max(1)
        lengths += 1
        lengths_sorted, idx_sorted = lengths.sort(0, descending=True)
        x_sorted = x[idx_sorted]
        y_sorted = None
        if y is not None:
            y_sorted = y[idx_sorted]
        lengths_list = lengths_sorted.numpy()
        return x_sorted, y_sorted, lengths_list, idx_sorted

    def _restore_order(self, x):
        if self.idx_sorted is not None:
            inv_idxs = self.idx_sorted.clone()
            inv_idxs.scatter_(0, self.idx_sorted, torch.arange(x.size(0)).long())
            return x[inv_idxs]
        return x

    def _to_var(self, x):
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def _to_numpy(self, x):
        return x.data.cpu().numpy().astype(float)