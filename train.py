import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datasets
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('run_name')
    parser.add_argument('--archive', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--repr-dims', type=int, default=320)
    parser.add_argument('--max-train-length', type=int, default=3000)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-threads', type=int, default=None)
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--irregular', type=float, default=0)
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    if args.archive == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datasets.load_UCR(args.dataset)
    elif args.archive == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datasets.load_UEA(args.dataset)
    elif args.archive == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datasets.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datasets.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datasets.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datasets.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
    else:
        assert False, 'unknown archive'
        
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            assert False
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)
    
    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.mkdir(run_dir)
    
    t = time.time()
    
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t

    print()
    print(f"Training time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)
        
    print()
    