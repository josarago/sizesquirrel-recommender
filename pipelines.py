from sklearn.preprocessing import (
    OrdinalEncoder,
    FunctionTransformer,
    LabelEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklego.preprocessing import ColumnSelector

from sklearn import set_config

set_config(transform_output="pandas")

TARGET_CATEGORIES = [1, 2, 3, 4, 5]
TARGET_COLUMN = "rating"
EMBEDDING_COLUMNS = dict()

EMBEDDING_COLUMNS["user"] = [
    "user_id",
    "user_gender",
    "user_foot_shape",
]

EMBEDDING_COLUMNS["sku"] = [
    "brand_id",
    "model",
    "shoe_gender",
    "size",
    "item_type",
]

embedding_pipe = Pipeline(
    steps=[
        (
            "embedding_columns",
            ColumnSelector(EMBEDDING_COLUMNS["user"] + EMBEDDING_COLUMNS["sku"]),
        ),
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

SIZE_IN_SCALE = 15

user_street_shoe_size_in_pipe = Pipeline(
    steps=[
        ("user_street_shoe_size_in", ColumnSelector("user_street_shoe_size_in")),
        ("imputer", SimpleImputer(fill_value=-SIZE_IN_SCALE)),
        ("scale", FunctionTransformer(lambda x: x / SIZE_IN_SCALE)),
    ]
)


CLIMBING_YEAR_IN_STYLE_COLUMNS = ["bouldering", "sport_climbing", "trad_climbing"]

climbing_years_in_style_pipe = Pipeline(
    steps=[
        ("column_selector", ColumnSelector(CLIMBING_YEAR_IN_STYLE_COLUMNS)),
        ("scaler", StandardScaler()),
    ]
)

user_features_pipe = FeatureUnion(
    [
        ("user_street_shoe_size", user_street_shoe_size_in_pipe),
        ("climbing_year_in_style", climbing_years_in_style_pipe),
    ]
)


SKU_NUMERICAL_COLUMNS = ["size_in"]

shoe_size_in_pipe = Pipeline(
    steps=[
        ("shoe_size_in", ColumnSelector(SKU_NUMERICAL_COLUMNS)),
        ("imputer", SimpleImputer(fill_value=-SIZE_IN_SCALE)),
        ("scale", FunctionTransformer(lambda x: x / SIZE_IN_SCALE)),
    ]
)

sku_features_pipe = FeatureUnion(
    [
        ("shoe_size", shoe_size_in_pipe),
    ]
)


classifier_target_pipe = LabelEncoder()
regressor_target_pipe = FunctionTransformer(lambda x: x)
