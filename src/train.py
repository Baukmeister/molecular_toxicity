# %% imports
import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset import MolecularToxicityDataset
from model import GNN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss / step


def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss / step


def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'data/images/cm_{epoch}.png')


def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_pred, y_true)}")
    print(f"Accuracy: {accuracy_score(y_pred, y_true)}")
    prec = precision_score(y_pred, y_true)
    rec = recall_score(y_pred, y_true)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    try:
        roc = roc_auc_score(y_pred, y_true)
        print(f"ROC AUC: {roc}")
    except:
        print(f"ROC AUC: notdefined")


# %% Run the training
def run_training(graphs, labels):

    # Loading the dataset
    print("Loading dataset...")
    train_dataset = MolecularToxicityDataset(root="../data/torch_data_set", graphs=graphs, labels=labels)
    test_dataset = MolecularToxicityDataset(root="../data/torch_data_set", graphs=graphs, labels=labels, test=True)

    # Prepare training
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Loading the model
    print("Loading model...")

    model_params = {
        "model_embedding_size": [64],
        "model_attention_heads": [3],
        "model_layers": [4],
        "model_dropout_rate": [0.2],
        "model_top_k_ratio": [0.5],
        "model_top_k_every_n": [1],
        "model_dense_neurons": [256],
        "model_edge_dim": [1]
    }

    model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)
    model = model.to(device)
    print(f"Number of parameters: {count_parameters(model)}")

    # < 1 increases precision, > 1 recall
    weight = torch.tensor([1.0], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.8,
                                weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # Start training
    best_loss = 1000
    early_stopping_counter = 0
    for epoch in range(300):
        if early_stopping_counter <= 10:  # = x * 5
            # Training
            model.train()
            loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
            print(f"Epoch {epoch} | Train Loss {loss}")

            # Testing
            model.eval()
            if epoch % 5 == 0:
                loss = test(epoch, model, test_loader, loss_fn)
                print(f"Epoch {epoch} | Test Loss {loss}")

                # Update best loss
                if float(loss) < best_loss:
                    best_loss = loss
                    # Save the currently best model
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            print("Early stopping due to no improvement.")
            return [best_loss]
    print(f"Finishing training with best test loss: {best_loss}")
    return [best_loss]