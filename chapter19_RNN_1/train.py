import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from mnist_classifier.models.cnn import ConvolutionalClassifier
from mnist_classifier.trainer import Trainer

from mnist_classifier.dataloader import load_mnist, split_data, get_hidden_sizes, get_model

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--model', default='fc', choices=['fc', 'cnn', 'rnn'])
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=0.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--early_stop', type=int, default=0)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=0.3)

    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config

def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')

    X, y = load_mnist(is_train=True, flatten=(config.model == 'fc'))
    X, y = split_data(X.to(device), y.to(device), train_ratio=config.train_ratio)

    print('Train:', X[0].shape, y[0].shape)
    print('Valid:', X[1].shape, y[1].shape)

    input_size = int(X[0].shape[-1])
    output_size = int(max(y[0])) + 1

    model = get_model(input_size, output_size, config).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model=model, optimizer=optimizer, crit=crit)

    trainer.train(train_data=(X[0], y[0]), valid_data=(X[1], y[1]), config=config)

    torch.save({'model':trainer.model.state_dict(),
                'opt':optimizer.state_dict(),
                'config':config}, config.model_fn)
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)