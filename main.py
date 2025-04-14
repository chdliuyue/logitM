import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_read import load_config, load_data, CustomDataset
from utils.select_model import select_model
from train import train_epoch, validate
from torch.utils.data import DataLoader
import numpy as np


def main(path='./configs/swissmetro.yml'):
    config = load_config(path)

    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, Y_train = load_data(config, set_name='train')
    X_test, Y_test = load_data(config, set_name='test')
    train_dataset = CustomDataset(X_train, Y_train)
    test_dataset = CustomDataset(X_test, Y_test)
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # train_loader = DataLoader(train_dataset, batch_size=X_train.shape[0], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=X_test.shape[0], shuffle=False)

    model = select_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    num_epochs = config['num_epochs']
    plot_loss = []
    plot_acc = []
    plot_F1 = []
    best_loss = float('inf')
    best_model = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_F1 = train_epoch(model, train_loader, criterion, optimizer, device)
        plot_loss.append(train_loss)
        plot_acc.append(train_acc)
        plot_F1.append(train_F1)
        if train_loss < best_loss:
            best_loss = train_loss
            best_model = model.state_dict()
        print(f"Epoch {epoch}/{num_epochs} Train Loss: {train_loss: .4f} Accuracy: {train_acc: .4f} F1: {train_F1: .4f}")

    model.load_state_dict(best_model)
    test_loss, test_acc, test_F1, test_precision, test_recall, conf_matrix, all_preds, all_targets, all_probs = \
        validate(model, test_loader, criterion, device)
    print(f"\n model selcted: {config['model_type']}\n")
    print(f"Test Loss: {test_loss: .4f} Accuracy: {test_acc: .4f} F1: {test_F1: .4f} "
          f"Precision: {test_precision: .4f} Recall: {test_recall: .4f} ")
    print(f"Confusion Matrix: \n{conf_matrix}")


if __name__ == "__main__":
    path = './configs/swissmetro.yml'
    main(path)

