import os
import argparse
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from model.blip2_stage1 import Blip2Stage1
from model.unimol import SimpleUniMolModel
from data_provider.stage1_dm import Stage1DM
from model.dist_funs import MyDeepSpeedStrategy
import numpy as np
from model.blip2_stage1 import test_model
import shap
os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = Blip2Stage1.load_from_checkpoint(args.init_checkpoint, device=args.devices, strict=False)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = Blip2Stage1(args)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # data
    dm = Stage1DM(args.num_workers, args.batch_size, args.root, args.text_max_len, model.blip2qformer.dictionary, model.blip2qformer.tokenizer, args)
    test = dm.test_match_loader
    model = model.blip2qformer
    acc, acc0, acc1, acc2, acc3, acc4 = test_model(model, test, model.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="stage1")

    parser.add_argument('--seed', type=int, default=114514, help='random seed')


    parser.add_argument('--gtm', action='store_false', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_false', help='use language modeling or not', default=False)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--use_3d', action='store_true', default=True)
    parser.add_argument('--enriched_descrption', action='store_true', default=True)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--save_every_n_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)    
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = Stage1DM.add_model_specific_args(parser)
    parser = SimpleUniMolModel.add_args(parser)
    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

