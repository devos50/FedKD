import os

from torch.utils.data import DataLoader

from dataset import data_cifar


def setup_dataset(args):
    args.datapath = os.path.expanduser(args.datapath)
    assert args.dataset == 'cifar10' or args.dataset == 'cifar100'
    publicdata = 'cifar100' if args.dataset == 'cifar10' else 'imagenet'
    args.N_class = 10 if args.dataset == 'cifar10' else 100
    priv_data, _, test_dataset, public_dataset, distill_loader = data_cifar.dirichlet_datasplit(
        args, privtype=args.dataset, publictype=publicdata, N_parties=args.N_parties, online=not args.oneshot,
        public_percent=args.public_percent)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, sampler=None)

    return priv_data, public_dataset, distill_loader, test_loader
