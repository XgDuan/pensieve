import time
import importlib
import torch
import os
from dataloader import construct_dataloader, map_b, map_h

from utils import set_device, get_logger, seq_metric

device = None


def train(model, train_loader, optimizer, epoch, logger, log_step):

    model.train()
    epoch_loss = 0
    epoch_token_accu = 0
    epoch_seq_accu = 0
    for idx, batch_data in enumerate(train_loader):
        batch_input, batch_target, category = batch_data
        loss, result = model.train_batch(batch_input, batch_target, category, optimizer, logger)
        l1_error, l2_error = seq_metric(result, batch_target)
        epoch_loss += loss
        epoch_token_accu += l1_error
        epoch_seq_accu += l2_error

        if idx % log_step == 0:
            logger.info('train: epoch[%05d], batch[%04d/%04d], loss: %06.6f, '
                        'l1_error: %06.6f, l2_error: %06.6f',
                        epoch, idx, len(train_loader), loss,
                        l1_error, l2_error)

    logger.infov('train: epoch[%05d], loss: %06.6f, '
                 'l1_error: %06.6f, l2_error: %06.6f',
                 epoch, epoch_loss / len(train_loader),
                 epoch_token_accu / len(train_loader),
                 epoch_seq_accu / len(train_loader))

    return epoch_token_accu / len(train_loader), epoch_seq_accu / len(train_loader)


def test(model, test_loader, epoch, logger, log_step):

    model.eval()
    epoch_token_accu = 0
    epoch_seq_accu = 0

    for idx, batch_data in enumerate(test_loader):
        batch_input, batch_target, category = batch_data
        result = model.infer_batch(batch_input, logger)
        l1_error, l2_error = seq_metric(result, batch_target)
        epoch_token_accu += l1_error
        epoch_seq_accu += l2_error
        if idx % log_step == 0:
            logger.info('test: epoch[%05d], '
                        'l1_error: %06.6f, l2_error: %06.6f',
                        epoch, l1_error, l2_error)

    logger.infov('test: epoch[%05d], l1_error: %06.6f, l2_error: %06.6f',
                 epoch, epoch_token_accu / len(test_loader),
                 epoch_seq_accu / len(test_loader))
    return epoch_token_accu / len(test_loader), epoch_seq_accu / len(test_loader)


def get_model_by_name(model_name):
    try:
        Model = importlib.import_module('models.%s_model' % model_name).Model
    except Exception, e:
        print(e)
        exit()
    return Model


def main(args):
    Model = get_model_by_name(args['model'])

    if args['device'] == 'cpu':
        logger = get_logger('%s(%s)' % (args['alias'], 'cpu'))
        set_device('cpu', logger)
        device = torch.device('cpu')
    elif args['device'] == 'gpu':
        assert torch.cuda.is_available(), 'Cuda not available, pls use cpu'
        logger = get_logger('%s(%s)' % (args['alias'],
                                        'gpu%d' % args['gpu_id']))
        set_device('cuda:%d' % args['gpu_id'], logger)
        device = torch.device('cuda:%d' % args['gpu_id'])
    if 'hsdpa' in args['data_path']:
        category_map = map_h
    elif 'belgium' in args['data_path']:
        category_map = map_b
    else:
        assert False
    train_loader = construct_dataloader(os.path.join(args['data_path'], 'train'),
                                        args['batch_size'], category_map, logger)
    test_loader = construct_dataloader(os.path.join(args['data_path'], 'test'),
                                       args['batch_size'], category_map, logger)

    model = Model(**args)
    model.to(device)
    logger.info(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    train_err_record = list()
    test_err_record = list()
    best_test_err = 100
    best_state_dict = None
    for epoch in range(args['max_epoch']):  # TODO: resolve
        train_err = train(model, train_loader, optimizer, epoch, logger, args['log_step'])
        test_err = test(model, test_loader, epoch, logger, args['log_step'])
        train_err_record.append((epoch, train_err[0], train_err[1]))
        test_err_record.append((epoch, test_err[0], test_err[1]))
        if test_err[0] < best_test_err:
            best_test_err = test_err[0]
            best_state_dict = model.state_dict()

    # save result
    with open('%s/%s_%s_%s_%s_%02.2f_%s.pkl'
              % (args['result_path'], args['model'],
                 args['data_path'].split('/')[-1], args['step'],
                 args['is_bidir'], best_test_err,
                 time.strftime("%m-%d-%H:%M:%S", time.localtime())), 'wb') as f:
        torch.save({'train_record': train_err_record,
                    'args': args,
                    'test_record': test_err_record,
                    'best_state_dict': best_state_dict}, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    ###########################################################################
    # system
    ###########################################################################
    parser.add_argument('--random_seed', type=int, default=233,
                        help='random seed used. Note: if you load models from '
                             'a checkpoint, the random seed would be invalid.')
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['cpu', 'gpu'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--alias', type=str, default='test')
    parser.add_argument('--log_step', type=int, default=20)
    ###########################################################################
    # folders
    ###########################################################################
    parser.add_argument('--data_path', type=str, default='../data/dataset_belgium')
    parser.add_argument('--result_path', type=str, default='./result')
    ###########################################################################
    # #####optimize
    ###########################################################################
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=50)
    ###########################################################################
    # models
    ###########################################################################
    parser.add_argument('--model', type=str,
                        help='the model to be used, most related parameters is'
                             ' assigned with the model', default='rnn')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--step', type=int, default=4)
    parser.add_argument('--encoder_layer', type=int, default=2)
    parser.add_argument('--is_bidir', type=bool, default=False)
    args = parser.parse_args()
    args = vars(args)
    main(args)
