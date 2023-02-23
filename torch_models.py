import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RegressorConfig, ClassifierConfig
from trainer import Trainer
from pipelines import EMBEDDING_COLUMNS


class RatingPredictor(torch.nn.Module):
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
            self.embeddings[name] = torch.nn.Embedding(
                vocab_size, embedding_dim, sparse=False
            )
            self.biases[name] = torch.nn.Embedding(vocab_size, 1, sparse=False)
        self.user_features_stack = torch.nn.Sequential(
            torch.nn.Linear(self.user_features_dim, self.embedding_dim, dtype=float),
            torch.nn.ReLU(),
        )

        self.sku_features_stack = torch.nn.Sequential(
            torch.nn.Linear(self.sku_features_dim, self.embedding_dim, dtype=float),
            torch.nn.ReLU(),
        )

    def forward(self, inputs: dict) -> torch.tensor:
        # embedding and biases
        x_embed_inputs = dict()
        x_bias_inputs = dict()
        for name in self.vocabularies.keys():
            x_embed_inputs[name] = torch.squeeze(
                self.embeddings[name](torch.Tensor(inputs[name].values).long()), dim=1
            )
            x_bias_inputs[name] = torch.squeeze(
                self.biases[name](torch.Tensor(inputs[name].values).long()), dim=1
            )
        # user
        # features
        x_features_user = self.user_features_stack(
            torch.tensor(inputs["user_features"].values)
        )
        # sum all embeddings
        x_embed_user = torch.sum(
            torch.stack(
                [
                    x_embed_inputs[name]
                    for name, x in inputs.items()
                    if name in EMBEDDING_COLUMNS["user"]
                ]
                + [x_features_user]
            ),
            dim=0,
        )
        # bias
        x_bias_user = torch.squeeze(
            torch.sum(
                torch.stack(
                    [
                        x_bias_inputs[name]
                        for name, x in inputs.items()
                        if name in EMBEDDING_COLUMNS["user"]
                    ]
                ),
                dim=0,
            ),
            dim=1,
        )

        # sku
        # features
        x_features_sku = self.sku_features_stack(
            torch.tensor(inputs["sku_features"].values)
        )
        # sum all sku embeddings
        x_embed_sku = torch.sum(
            torch.stack(
                [
                    x_embed_inputs[name]
                    for name, x in inputs.items()
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
                        for name, x in inputs.items()
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
            # create a tensor fo size (batch_size, 1, embedding_dim)
            raise ValueError("fix dimensions")
            # combined = torch.subtract(x_embed_user, x_embed_sku)
        else:
            raise ValueError("dot or subtract")
        with_biases = torch.sum(torch.stack([x_bias_user, x_bias_sku, combined]), dim=0)
        return with_biases


if __name__ == "__main__":
    model_config = RegressorConfig(
        fit_verbose=0,
        learning_rate=0.0001,
        epochs=1_000,
        embedding_dim=8,
        batch_size=1024,
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

    user_features_dim = inputs_train["user_features"].shape[1]
    sku_features_dim = inputs_train["sku_features"].shape[1]

    rating_predictor = RatingPredictor(
        7, embedding_vocabs, user_features_dim, sku_features_dim
    )
    print(rating_predictor(inputs_train).shape)