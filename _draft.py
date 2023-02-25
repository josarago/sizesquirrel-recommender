from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import optim


from config import RegressorConfig, ClassifierConfig
from trainer import Trainer
from pipelines import EMBEDDING_COLUMNS
from torch_models import RatingPredictor
from datasets import SizeSquirrelRecommenderDataset


def get_history_df(n_folds, n_epochs):
    nan_array = np.empty((n_epochs, n_folds))
    nan_array[:] = np.nan
    df = pd.DataFrame(
        data=nan_array,
        index=range(n_epochs),
        columns=[f"fold{n+1}" for n in range(n_folds)],
    )

    df.index.rename("epoch", inplace=True)
    return df


def train_one_epoch(model, optimizer, loss_fn, training_loader):
    sum_loss = 0.0
    for batch_idx, batch_data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, targets = batch_data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        sum_loss += loss.item()
    epoch_mean_loss = sum_loss / (batch_idx + 1)
    return epoch_mean_loss


def validate_one_epoch(model, validation_loader, loss_fn):
    sum_loss = 0.0
    for batch_idx, batch_data in enumerate(validation_loader):
        # Every data instance is an input + label pair
        inputs, targets = batch_data
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss
        loss = loss_fn(outputs, targets)
        # Gather data and report
        sum_loss += loss.item()
    epoch_mean_loss = sum_loss / (batch_idx + 1)
    return epoch_mean_loss


# DEVICE = get_default_inference_device()
DEVICE = torch.device("cpu")

if __name__ == "__main__":
    model_config = RegressorConfig(
        fit_verbose=0,
        learning_rate=0.0001,
        epochs=100,
        embedding_dim=4,
        batch_size=256,
        embedding_func="subtract",
        early_stopping__restore_best_weights=False,
    )
    trainer = Trainer(model_config)
    # load data
    trainer.load_data()
    trainer.transform_data()
    df_train, df_test = trainer.get_split_training_set()
    trainer.fit_pipelines(df_train)
    inputs_train, embedding_vocabs = trainer.get_inputs_dict(df_train)
    targets_train = trainer.get_targets(df_train)
    history = {
        key: get_history_df(n_folds=5, n_epochs=trainer.model_config.epochs)
        for key in ["train_loss", "val_loss", "train_mae", "val_mae"]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4321)
    loss_fn = torch.nn.MSELoss()
    for fold, (train_idxs, val_idxs) in enumerate(
        skf.split(targets_train, targets_train)
    ):

        user_features_dim = inputs_train["user_features"].shape[1]
        sku_features_dim = inputs_train["sku_features"].shape[1]

        print(f"Fold #{fold + 1}")

        train_dataset = SizeSquirrelRecommenderDataset(
            inputs_train, targets_train, idxs=train_idxs, device=DEVICE
        )
        val_dataset = SizeSquirrelRecommenderDataset(
            inputs_train, targets_train, idxs=val_idxs, device=DEVICE
        )
        train_loader = DataLoader(
            train_dataset, batch_size=trainer.model_config.batch_size
        )
        val_loader = DataLoader(val_dataset, batch_size=trainer.model_config.batch_size)

        model = RatingPredictor(
            trainer.model_config.embedding_dim,
            embedding_vocabs,
            user_features_dim,
            sku_features_dim,
        ).to(DEVICE)
        #     model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        with tqdm(range(trainer.model_config.epochs), mininterval=2) as epoch_bar:
            for epoch in epoch_bar:
                epoch_mean_train_loss = train_one_epoch(
                    model, optimizer, loss_fn, train_loader
                )
                epoch_mean_val_loss = validate_one_epoch(model, val_loader, loss_fn)
                epoch_bar.set_description(
                    f"training loss: {epoch_mean_train_loss:.2f} - validation loss: {epoch_mean_val_loss:.2f}"
                )
