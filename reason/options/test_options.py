from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """Test Option Class"""

    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--load_checkpoint_path', required=True, type=str, help='checkpoint path')
        self.parser.add_argument('--save_result_path', required=True, type=str, help='save result path')
        self.parser.add_argument('--max_val_samples', default=None, type=int, help='max val data')
        self.parser.add_argument('--batch_size', default=256, type=int, help='batch_size')

        self.is_train = False