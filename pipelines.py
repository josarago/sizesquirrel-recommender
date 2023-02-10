from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

embedding_pipe = Pipeline(
    steps=[
        (
            "ordinal_encoder",
            OrdinalEncoder(
                categories="auto",
                dtype=int,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            ),
        )
    ]
).set_output(transform="pandas")
