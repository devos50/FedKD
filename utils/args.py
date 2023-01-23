import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=8)
    # data
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--datapath', type=str, default='/data_local/xuangong/data/')
    # path
    parser.add_argument('--logfile', default='', type=str)
    parser.add_argument('--subpath', type=str, default='')  # subpath under localtraining folder to save central model
    parser.add_argument('--checkpoint_path', type=str, default='')

    # local training
    parser.add_argument('--N_parties', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--C', type=float, default=1)  # percent of locals selected in each fed communication round
    parser.add_argument('--fedinitepochs', type=int, default=20)  # epochs of local training
    parser.add_argument('--batchsize', type=int, default=16)  # 128
    parser.add_argument('--lr', type=float, default=0.0025)  # 0.025
    parser.add_argument('--lrmin', type=float, default=0.001)
    parser.add_argument('--distill_droprate', type=float, default=0)
    parser.add_argument('--optim', type=str, default='ADAM')
    parser.add_argument('--ltparallel', type=int, default=None)
    parser.add_argument('--cindex', type=int, default=None)  # Client index, used for parallel local training
    parser.add_argument('--das', action='store_true')
    parser.add_argument('--dasclients', type=str, default='')

    # fed setting
    parser.add_argument('--fedrounds', type=int, default=200)
    parser.add_argument('--public_percent', type=float, default=1.0)  # ablation for c100 as public data
    parser.add_argument('--oneshot', action='store_true')
    parser.add_argument('--joint', action='store_true')  # only valid when wo/ QN
    parser.add_argument('--quantify', type=float, default=0.0)  # when w/ QN
    parser.add_argument('--noisescale', type=float, default=0.0)  # when w/ QN

    # fed training param
    parser.add_argument('--disbatchsize', type=int, default=512)
    parser.add_argument('--localepochs', type=int, default=10)
    parser.add_argument('--initepochs', type=int, default=500)
    parser.add_argument('--initcentral', type=str,
                        default='')  # ckpt used to init central model, import for co-distillation
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--steps_round', type=int, default=10000)
    parser.add_argument('--dis_lr', type=float, default=1e-3)  # 1e-3
    parser.add_argument('--dis_lrmin', type=float, default=1e-3)  # 1e-5
    parser.add_argument('--momentum', type=float, default=0.9)

    # ensemble
    parser.add_argument('--voteout', action='store_true')
    parser.add_argument('--clscnt', type=int, default=1)  # local weight specific to class
    # loss
    parser.add_argument('--lossmode', type=str, default='l1')  # kl or l1

    return parser.parse_args()