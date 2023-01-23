import socket

from tensorboardX import SummaryWriter

from datetime import datetime

import models.resnet8 as resnet8
from dataset import setup_dataset

from utils.args import get_args
from utils.myfed import *

if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    args.can_gpu = torch.cuda.is_available()
    
    handlers = [logging.StreamHandler()]
    if args.logfile:
        args.logfile = f'{datetime.now().strftime("%m%d%H%M")}'+args.logfile
    else:
        if args.joint:
            args.logfile = f'{datetime.now().strftime("%m%d%H%M")}_joint_a{args.alpha}s{args.seed}c{args.C}-sr{args.steps_round}'
        else:
            args.logfile = f'{datetime.now().strftime("%m%d%H%M")}_{args.dataset}_a{args.alpha}s{args.seed}c{args.C}_os{args.oneshot}_q{args.quantify}n{args.noisescale}'
    
    writer = SummaryWriter(comment=args.logfile)
    if not os.path.isdir('./logs'):
        os.mkdir('./logs')
    if args.debug:
        writer = None
        handlers.append(logging.FileHandler(
            f'./logs/debug.txt', mode='a'))
    else:
        handlers.append(logging.FileHandler(
            f'./logs/{args.logfile}.txt', mode='a'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    logging.info(args)
    
    # 1. Setup datasets
    priv_data, public_dataset, distill_loader, test_loader = setup_dataset(args)
    
    ###########
    # 2. model
    logging.info("CREATE MODELS.......")
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    logging.info(f'GPU: {args.gpu}')
    model = resnet8.ResNet8(num_classes=args.N_class)
    if args.can_gpu:
        model = model.cuda()
    logging.info("totally {} paramerters".format(sum(x.numel() for x in model.parameters())))
    logging.info("Param size {}".format(np.sum(np.prod(x.size()) for name,x in model.named_parameters() if 'linear2' not in name)))
    if len(gpu)>1:
        model = nn.DataParallel(model, device_ids=gpu)
    
    # 3. fed training
    if args.quantify or args.noisescale:
        fed = FedKDwQN(model, distill_loader, priv_data, test_loader, writer, args)
        fed.init_locals()
        if args.oneshot:
            fed.update_distill_loader_wlocals(public_dataset)
            fed.distill_local_central_oneshot()
        # else:
        #     fed.distill_local_central()
    else:
        fed = FedKD(model, distill_loader, priv_data, test_loader, writer, args)
        fed.init_locals()
        if args.joint:
            fed.distill_local_central_joint()
        elif args.oneshot:
            fed.update_distill_loader_wlocals(public_dataset)
            fed.distill_local_central_oneshot()
        # else:
        #     fed.distill_local_central()
    
    if not args.debug:
        writer.close()
