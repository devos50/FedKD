"""
Stand-alone script to train a particular local model.
"""
import logging
import os

import torch.cuda
from tensorboardX import SummaryWriter

from dataset import setup_dataset
from models import resnet8
from utils.args import get_args
from utils.myfed import FedKD

if __name__ == "__main__":
    args = get_args()

    handlers = [logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format='<subprocess> %(asctime)s - %(levelname)s - %(message)s', handlers=handlers,
    )

    args.can_gpu = torch.cuda.is_available()
    writer = SummaryWriter(comment="%d" % args.cindex)

    args.N_class = 10 if args.dataset == 'cifar10' else 100
    model = resnet8.ResNet8(num_classes=args.N_class)
    if args.can_gpu:
        model = model.cuda()

    logging.info("Subprocess for client %d started" % args.cindex)

    # Set up the datasets
    priv_data, public_dataset, distill_loader, test_loader = setup_dataset(args)

    fed = FedKD(model, distill_loader, priv_data, test_loader, writer, args)
    savename = os.path.join(fed.rootdir, str(args.cindex) + '.pt')
    acc = fed.trainLocal(savename, modelid=args.cindex)

    logging.info("Achieved accuracy for model %d: %f", args.cindex, acc)
