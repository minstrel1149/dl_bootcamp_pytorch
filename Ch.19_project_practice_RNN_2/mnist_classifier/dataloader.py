import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from mnist_classifier.models.model_fc import ImageClassifier
from mnist_classifier.models.model_cnn import ConvolutionClassifier
from mnist_classifier.models.model_rnn import SequenceClassifier

def load_mnist(is_train=True, download=False, flatten=True):
    dataset = datasets.MNIST('../../data', train=is_train,
                             download=download,
                             transform=transforms.Compose([transforms.ToTensor()]))
    
    X = dataset.data.float() / 255
    y = dataset.targets

    if flatten:
        X = X.reshape(X.shape[0], -1)
    
    return X, y

def split_data(X, y, train_ratio=0.8):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1-train_ratio, random_state=0)

    return X_train, X_val, y_train, y_val

def get_model(input_size, output_size, config):
    if config.model == 'fc':
        model = ImageClassifier(
            input_size=input_size, output_size=output_size,
            n_layers=config.n_layers,
            use_batch_norm=not config.use_dropout,
            dropout_p=config.dropout_p
        )
    elif config.model == 'cnn':
        model = ConvolutionClassifier(
            output_size=output_size,
            image_size=config.image_size,
            base_channels=config.base_channels
        )
    elif config.model == 'rnn':
        model = SequenceClassifier(
            input_size=input_size, hidden_size=config.hidden_size, output_size=output_size,
            n_layers=config.n_layers, dropout_p=config.dropout_p
        )
    else:
        raise NotImplementedError
    
    return model