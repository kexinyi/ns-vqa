from options.train_options import TrainOptions
from datasets import get_dataloader
from executors import get_executor
from models.parser import Seq2seqParser
from trainer import Trainer


opt = TrainOptions().parse()
train_loader = get_dataloader(opt, 'train')
val_loader = get_dataloader(opt, 'val')
model = Seq2seqParser(opt)
executor = get_executor(opt)
trainer = Trainer(opt, train_loader, val_loader, model, executor)

trainer.train()