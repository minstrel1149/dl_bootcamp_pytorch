import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from dataloader import load_mnist, split_data

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--model', default='fc', choices=['fc', 'cnn', 'rnn'])
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=0.8)

    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--early_stop', type=int, default=50)

    p.add_argument('--n_layers', type=int, default=7)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=0.2)

    p.add_argument('--clf', type=bool, default=True)

    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config

def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')

    X, y = load_mnist(is_train=True, flatten=(config.model == 'fc'))
    X_train, X_val, y_train, y_val = split_data(X.to(device), y.to(device), train_ratio=config.train_ratio)

    print('Train:', X_train.shape, y_train.shape)
    print('Valid:', X_val.shape, y_val.shape)

    input_size = int(X_train.shape[-1])
    output_size = int(max(y_train)) + 1

    model = ImageClasifier(input_size=input_size, output_size=output_size,
                           n_layers=config.n_layers,
                           use_batch_norm=not config.use_dropout,
                           dropout_p=config.dropout_p)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss() if config.clf is True else nn.MSELoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)
    
    trainer = Trainer(model=model, optimizer=optimizer, crit=crit)

    trainer.train(train_data=(X_train, y_train), valid_data=(X_val, y_val), config=config)

    torch.save({'model':trainer.model.state_dict(),
                'opt':optimizer.state_dict(),
                'config':config}, config.model_fn)
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)