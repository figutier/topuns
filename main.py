from data import UnraveledDataset
from data import load_data
from models import MLP
from train import *
import pickle
from torchvision import datasets
from utils_tda import *


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
    print(device)
    dataclass = UnraveledDataset
    dataset = datasets.MNIST
    batch_size = 1500
    dataloader_train, dataloader_val = load_data(dataclass, dataset, download = True)
    kwargs = dict(
        n_features=28 * 28,
        hidden_layer_sizes=(100, 100),
        n_targets=10
    )
    loss_inst = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
    model = MLP(**kwargs)
    model.to(device)
    train(model, dataloader_train, dataloader_val, loss = loss_inst, optimizer = optimizer, device = device)
    batch = next(iter(dataloader_train))
    n_obs_per_class = 100

    labels = np.unique(batch[1])
    # Take n_obs_per_class observation from each class in x_train:
    xs = [batch[0][np.where(batch[1] == label)][:n_obs_per_class] for label in labels]

    # Now we get the corresponding barycenters.
    all_barycenters = [barycenters_of_set_from_deep_model(model, x, layers_id= None) for x in xs]

    with open('try.pickle', 'wb') as handle:
        pickle.dump(all_barycenters, handle, protocol=pickle.HIGHEST_PROTOCOL)
