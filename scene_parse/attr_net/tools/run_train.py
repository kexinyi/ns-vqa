from options import get_options
from datasets import get_dataloader
from model import get_model
from trainer import get_trainer


opt = get_options('train')
train_loader = get_dataloader(opt, 'train')
val_loader = get_dataloader(opt, 'val')
model = get_model(opt)
trainer = get_trainer(opt, model, train_loader, val_loader)

trainer.train()