from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """Train option class"""

    def __init__(self):
        super(TrainOptions, self).__init__()
        # Data
        self.parser.add_argument('--max_train_samples', default=None, type=int, help='max number of training samples')
        self.parser.add_argument('--max_val_samples', default=10000, type=int, help='max number of val samples')
        # Model
        self.parser.add_argument('--load_checkpoint_path', default=None, type=str, help='checkpoint path')
        self.parser.add_argument('--encoder_max_len', default=50, type=int, help='max length of input sequence')
        self.parser.add_argument('--decoder_max_len', default=27, type=int, help='max length of output sequence')
        self.parser.add_argument('--hidden_size', default=256, type=int, help='hidden layer dimension')
        self.parser.add_argument('--word_vec_dim', default=300, type=int, help='dimension of word embedding vector')
        self.parser.add_argument('--input_dropout_p', default=0, type=float, help='dropout probability for input sequence')
        self.parser.add_argument('--dropout_p', default=0, type=float, help='dropout probability for output sequence')
        self.parser.add_argument('--n_layers', default=2, type=int, help='number of hidden layers')
        self.parser.add_argument('--rnn_cell', default='lstm', type=str, help='encoder rnn cell type, options: lstm, gru')
        self.parser.add_argument('--bidirectional', default=1, type=int, help='bidirectional encoder')
        self.parser.add_argument('--variable_lengths', default=1, type=int, help='variable input length')
        self.parser.add_argument('--use_attention', default=1, type=int, help='use attention in decoder')
        self.parser.add_argument('--use_input_embedding', default=0, type=int, help='use pretrained word embedding for input sentences')
        self.parser.add_argument('--fix_input_embedding', default=0, type=int, help='fix word embedding for input sentences')
        self.parser.add_argument('--start_id', default=1, type=int, help='id for start token')
        self.parser.add_argument('--end_id', default=2, type=int, help='id for end token')
        self.parser.add_argument('--null_id', default=0, type=int, help='id for null token')
        self.parser.add_argument('--word2vec_path', default=None, type=str, help='pretrained embedding path')
        self.parser.add_argument('--fix_embedding', default=0, type=int, help='fix pretrained embedding')
        # Training
        self.parser.add_argument('--reinforce', default=0, type=int, help='train reinforce')
        self.parser.add_argument('--batch_size', default=64, type=int, help='batch size')
        self.parser.add_argument('--learning_rate', default=7e-4, type=float, help='learning rate')
        self.parser.add_argument('--entropy_factor', default=0.0, type=float, help='entropy weight in reinforce loss')
        self.parser.add_argument('--num_iters', default=20000, type=int, help='total number of iterations')
        self.parser.add_argument('--reward_decay', default=0.9, type=float, help='decay weight for reward moving average')
        self.parser.add_argument('--display_every', default=20, type=int, help='display every')
        self.parser.add_argument('--checkpoint_every', default=1000, type=int, help='validate and save checkpoint every')
        self.parser.add_argument('--visualize_training', default=0, type=int, help='visualize training with tensorboard')

        self.is_train = True