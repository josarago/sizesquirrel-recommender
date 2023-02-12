from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklego.preprocessing import ColumnSelector

from sklearn import set_config

set_config(transform_output="pandas")

EMBEDDING_COLUMNS = ["user_id", "sku_id"]
TARGET_COLUMNS = ["rating"]

embedding_pipe = Pipeline(
    steps=[
        ("embedding_columns", ColumnSelector(EMBEDDING_COLUMNS)),
        (
            "ordinal_encoder",
            OrdinalEncoder(
                categories="auto",
                dtype=int,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            ),
        ),
    ]
)


target_pipe = Pipeline(
    steps=[
        ("embedding_columns", ColumnSelector(TARGET_COLUMNS)),
        ("one_hot_enc", OneHotEncoder(sparse_output=False)),
    ]
)
