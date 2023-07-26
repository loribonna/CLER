import numpy as np
import importlib
import os
import sys
import socket

conf_path = os.getcwd()
sys.path.append(conf_path)
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser, Namespace
from utils.args import add_management_args
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.conf import set_random_seed
from utils.pretext import get_pretext_args

import torch

import uuid
import datetime
import time

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

def merge_dataset_args(args):
    dataset = get_dataset(args)
    
    # merge args from dataset
    dataset_args = dataset.get_setting()
    # ugly fix, please ignore
    if "ewc" in args.model or "sgd" in args.model  or "join" in args.model or 'lwf' in args.model or 'rdfcil' in args.model:
        if hasattr(dataset_args, "minibatch_size"):
            del dataset_args.minibatch_size
        if hasattr(dataset_args, "buffer_size"):
            del dataset_args.buffer_size
    
    sys.argv = sys.argv[:1] + [f"--{k}={v}" for k, v in vars(dataset_args).items() if type(v) != list] + sys.argv[1:]
    # HANDLE OPT STEPS SEPARATELY
    for k, v in vars(dataset_args).items():
        if type(v) == list:
            sys.argv = sys.argv[:1] + ([f"--{k}"] + [str(x) for x in v]) + sys.argv[1:]
    return Namespace(**{**vars(dataset_args), **{k:v for k,v in vars(args).items() if v is not None or k not in dataset_args}})

def parse_args():
    print("------- ARGV -------\n",sys.argv,"\n---------------------\n", file=sys.stderr)
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    add_management_args(parser)
    parser = get_pretext_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    args = parser.parse_known_args()[0]
    args = merge_dataset_args(args)

    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    parser = get_pretext_args(parser)
    
    args = parser.parse_args()
    
    if args.seed is not None:
        set_random_seed(args.seed)

    # job number 
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_git_commit = os.popen(r"git log | head -1 | sed s/'commit '//").read().strip()

    args.conf_host = socket.gethostname()

    if args.savecheck:
        now = time.strftime("%Y%m%d-%H%M%S")
        args.ckpt_name = f"{args.model}_{args.dataset}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.n_epochs}_{str(now)}"
        args.ckpt_name_replace = f"{args.model}_{args.dataset}_{'{}'}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.n_epochs}_{str(now)}"

    return args

def main(args=None):
    torch.set_num_threads(4)
    start_time = time.time()

    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    dataset = get_dataset(args)

    backbone = dataset.get_backbone()

    loss = dataset.get_loss()

    if args.model == 'joint':
        args.ignore_other_metrics=1
    model = get_model(args, backbone, loss, dataset.get_transform())
    
    train(model, dataset, args) 

    if args.timeme:
        print('Total time: {}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
