from torch.utils.data import DataLoader
from .clevr_object import ClevrObjectDataset


def get_dataset(opt, split):
    if opt.dataset == 'clevr':
        if split == 'train':
            ds = ClevrObjectDataset(opt.clevr_mini_ann_path, opt.clevr_mini_img_dir, 'mini', 
                                    max_img_id=opt.split_id, concat_img=opt.concat_img)
        elif split == 'val':
            ds = ClevrObjectDataset(opt.clevr_mini_ann_path, opt.clevr_mini_img_dir, 'mini',
                                    min_img_id=opt.split_id, concat_img=opt.concat_img)
        elif split == 'test':
            ds = ClevrObjectDataset(opt.clevr_val_ann_path, opt.clevr_val_img_dir, 'val', concat_img=opt.concat_img)
        else:
            raise ValueError('Invalid dataset split: %s' % split)
    else:
        raise ValueError('Invalid datsaet %s' % opt.dataset)
    return ds


def get_dataloader(opt, split):
    ds = get_dataset(opt, split)
    loader = DataLoader(dataset=ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=opt.shuffle_data)
    return loader