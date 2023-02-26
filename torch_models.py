from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import pytorch_lightning as pl

from config import RegressorConfig
from pipelines import TARGET_CATEGORIES, EMBEDDING_COLUMNS


class RatingScaler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return (len(TARGET_CATEGORIES) - 1) * nn.Sigmoid()(input) + 1


def get_history_df(n_folds, n_epochs):
    nan_array = np.empty((n_epochs, n_folds))
    nan_array[:] = np.nan
    df = pd.DataFrame(
        data=nan_array,
        index=range(n_epochs),
        columns=[f"fold_{n+1}" for n in range(n_folds)],
    )
    df.index.rename("epoch", inplace=True)
    return df


class RatingRegressor(nn.Module):
    def __init__(
        self,
        embedding_dim,
        vocabularies,
        user_features_dim,
        sku_features_dim,
        combine_func="dot",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocabularies = vocabularies
        self.user_features_dim = user_features_dim
        self.sku_features_dim = sku_features_dim
        self.combine_func = combine_func
        # initialize
        self.embeddings = dict()
        self.biases = dict()
        # embedding inputs
        for name, vocabulary in vocabularies.items():
            vocab_size = len(vocabulary)
            self.embeddings[name] = nn.Embedding(
                vocab_size, embedding_dim, sparse=False
            )
            self.biases[name] = nn.Embedding(vocab_size, 1, sparse=False)
        self.user_features_stack = nn.Sequential(
            nn.Linear(self.user_features_dim, self.embedding_dim, dtype=float),
            nn.ReLU(),
        )

        self.sku_features_stack = nn.Sequential(
            nn.Linear(self.sku_features_dim, self.embedding_dim, dtype=float),
            nn.ReLU(),
        )
        self.rating_scaler = RatingScaler()

    def forward(self, inputs: dict) -> torch.tensor:
        # embedding and biases
        x_embed_inputs = dict()
        x_bias_inputs = dict()
        for name in self.vocabularies.keys():
            x_embed_inputs[name] = torch.squeeze(
                self.embeddings[name](inputs[name]), dim=1
            )
            x_bias_inputs[name] = torch.squeeze(self.biases[name](inputs[name]), dim=1)
        # user
        # features
        x_features_user = self.user_features_stack(inputs["user_features"])
        # sum all embeddings
        x_embed_user = torch.sum(
            torch.stack(
                [
                    x_embed_inputs[name]
                    for name in inputs.keys()
                    if name in EMBEDDING_COLUMNS["user"]
                ]
                + [x_features_user],
            ),
            dim=0,
        )
        # bias
        x_bias_user = torch.squeeze(
            torch.sum(
                torch.stack(
                    [
                        x_bias_inputs[name]
                        for name in inputs.keys()
                        if name in EMBEDDING_COLUMNS["user"]
                    ]
                ),
                dim=0,
            ),
            dim=1,
        )

        # sku
        # features
        x_features_sku = self.sku_features_stack(inputs["sku_features"])
        # sum all sku embeddings
        x_embed_sku = torch.sum(
            torch.stack(
                [
                    x_embed_inputs[name]
                    for name in inputs.keys()
                    if name in EMBEDDING_COLUMNS["sku"]
                ]
                + [x_features_sku]
            ),
            dim=0,
        )
        # bias
        x_bias_sku = torch.squeeze(
            torch.sum(
                torch.stack(
                    [
                        x_bias_inputs[name]
                        for name in inputs.keys()
                        if name in EMBEDDING_COLUMNS["sku"]
                    ]
                ),
                dim=0,
            ),
            dim=1,
        )
        if self.combine_func == "dot":
            # create a tensor fo size (batch_size, 1)
            combined = torch.mul(x_embed_user, x_embed_sku).sum(dim=1)
        elif self.combine_func == "subtract":
            subtracted = torch.subtract(x_embed_user, x_embed_sku).sum(dim=1)
            combined = torch.square(subtracted).sum(dim=1)
            raise ValueError("fix dimensions")
            # combined = torch.subtract(x_embed_user, x_embed_sku)
        else:
            raise ValueError("dot or subtract")
        with_biases = torch.sum(torch.stack([x_bias_user, x_bias_sku, combined]), dim=0)
        out = self.rating_scaler(with_biases)
        return out


class LitRatingRegressor(pl.LightningModule):
    def __init__(
        self,
        model_config: RegressorConfig,
        embedding_vocabs,
        user_features_dim,
        sku_features_dim,
    ):
        super().__init__()
        self.model_config = model_config
        self.model = RatingRegressor(
            self.model_config.embedding_dim,
            embedding_vocabs,
            user_features_dim,
            sku_features_dim,
        ).to(self.model_config.device)
        self.loss_fn = model_config.loss_fn
        self.learning_rate = model_config.learning_rate

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        output = self.model(x)
        loss = self.model_config.loss_fn(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        output = self.model(x)
        loss = self.model_config.loss_fn(output, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.model_config.learning_rate
        )
        return optimizer
