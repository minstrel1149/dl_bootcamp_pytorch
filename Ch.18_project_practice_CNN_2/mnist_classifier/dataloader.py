import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

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