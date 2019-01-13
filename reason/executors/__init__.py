from .clevr_executor import ClevrExecutor


def get_executor(opt):
    print('| creating %s executor' % opt.dataset)
    if opt.dataset == 'clevr':
        train_scene_json = opt.clevr_train_scene_path
        val_scene_json = opt.clevr_val_scene_path
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    executor = ClevrExecutor(train_scene_json, val_scene_json, vocab_json)
    return executor