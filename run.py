import argparse
import torch
from exp.exp_main import Exp_Main
import random
import time
import numpy as np


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def main():
    fix_seed = 0
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(
        description='Physics-Informed Mobility Perception Networks for Origin-Destination Flow Prediction and Human Mobility Interpretation')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='PIMPN',
                        help='model name, options: [PIMPN , , ]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/xc-lpr/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='od.npy', help='data file, options: [xc_od.npy, ]')
    parser.add_argument('--adj_path', type=str, default='adj_mx.pkl', help='adj data path')
    parser.add_argument('--poi_path', type=str, default='poi_type_counts.csv', help='poi data path')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # model define
    parser.add_argument('--num_nodes', type=int, default=333, help='number of nodes/variables')
    parser.add_argument('--num_pairs', type=int, default=378, help='number of valid od pairs')
    parser.add_argument('--num_categories', type=int, default=49, help='number of categories')
    parser.add_argument('--n_layers', type=int, default=2, help='num of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--sc_true', type=str_to_bool, default=True, help='whether to add Spatial encoder')
    parser.add_argument('--buildA_true', type=str_to_bool, default=True,
                        help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--use_poi', type=str_to_bool, default=True, help='whether to use poi information')
    parser.add_argument('--use_phy', type=str_to_bool, default=True, help='whether to use physical diffusion process')
    parser.add_argument('--use_att', type=str_to_bool, default=True, help='whether to use attention')
    parser.add_argument('--res_channels', type=int, default=64, help='res channels')
    parser.add_argument('--end_channels', type=int, default=512, help='end channels')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')  #
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer learning rate')
    parser.add_argument('--lambda_', type=float, default=5e-3, help='phy loss rate')
    parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')
    parser.add_argument('--cl_steps', type=int, default=100, help='curriculum learning steps')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_sl{}_pl{}_eb{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.seq_len,
                args.pred_len,
                args.embed,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_sl{}_pl{}_eb{}_{}_{}'.format(args.model_id,
                                                      args.model,
                                                      args.seq_len,
                                                      args.pred_len,
                                                      args.embed,
                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The code execution time is: {execution_time} s")
