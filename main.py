import os
from transformers import logging
import torch
import argparse
import logging as log
import Deberta_NLI

logging.set_verbosity_error()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--resume', nargs='?', const=True, default=False)
    parser.add_argument('--policy', type=str)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--only-test', nargs='?', const=True, default=False)
    parser.add_argument('--only-infer', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.policy not in ['Deberta_NLI']:
        print("--policy should be in ['Deberta_NLI']")
        exit(0)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.only_infer:
        eval(args.policy + '.inferring(batch_size={})'.format(args.batch_size))
        exit(0)

    logger = log.getLogger()
    logger.setLevel(log.INFO)

    if not os.path.exists('./res'):
        os.makedirs('./res')

    if not os.path.exists('./res/' + args.policy):
        os.makedirs('./res/' + args.policy)

    file_handler = log.FileHandler('./res/' + args.policy + '/train.log')
    file_handler.setFormatter(log.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = log.StreamHandler()
    stream_handler.setFormatter(log.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    log.info('------------------------------------------------')
    log.info("device: {}".format(device))
    log.info("epochs: {}  batch-size: {}  resume: {}".format(args.epochs, args.batch_size, args.resume))
    log.info('------------------------------------------------')

    if not args.only_test:
        eval(args.policy + '.train(EPOCHS={}, batch_size={}, lr={}, resume={})'.format(args.epochs, args.batch_size, args.lr, args.resume))

    eval(args.policy + '.test()')
