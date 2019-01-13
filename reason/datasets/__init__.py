from .clevr_questions import ClevrQuestionDataset
from torch.utils.data import DataLoader


def get_dataset(opt, split):
    """Get function for dataset class"""
    assert split in ['train', 'val']

    if opt.dataset == 'clevr':
        if split == 'train':
            question_h5_path = opt.clevr_train_question_path
            max_sample = opt.max_train_samples
        else:
            question_h5_path = opt.clevr_val_question_path
            max_sample = opt.max_val_samples
        dataset = ClevrQuestionDataset(question_h5_path, max_sample, opt.clevr_vocab_path)
    else:
        raise ValueError('Invalid dataset')

    return dataset


def get_dataloader(opt, split):
    """Get function for dataloader class"""
    dataset = get_dataset(opt, split)
    shuffle = opt.shuffle if split == 'train' else 0
    loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.num_workers)
    print('| %s %s loader has %d samples' % (opt.dataset, split, len(loader.dataset)))
    return loader