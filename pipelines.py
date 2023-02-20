from sklearn.preprocessing import (
    OrdinalEncoder,
    FunctionTransformer,
    OneHotEncoder,
    LabelBinarizer,
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklego.preprocessing import ColumnSelector

from sklearn import set_config

set_config(transform_output="pandas")

EMBEDDING_COLUMNS = ["user_id", "sku_id"]
TARGET_CATEGORIES = [1, 2, 3, 4, 5]
TARGET_COLUMN = "rating"

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

SIZE_IN_SCALE = 15

user_street_shoe_size_in_pipe = Pipeline(
    steps=[
        ("user_street_shoe_size_in", ColumnSelector("street_shoe_size_in")),
        ("imputer", SimpleImputer(fill_value=-SIZE_IN_SCALE)),
        ("scale", FunctionTransformer(lambda x: x / SIZE_IN_SCALE)),
    ]
)

USER_CATEGORICAL_COLUMNS = ["user_gender", "user_foot_shape"]

user_categorical_pipe = Pipeline(
    steps=[
        ("user_categories", ColumnSelector(USER_CATEGORICAL_COLUMNS)),
        ("one_hot", OneHotEncoder(drop="first", sparse_output=False)),
    ]
)

user_features_pipe = FeatureUnion(
    [
        ("user_street_shoe_size", user_street_shoe_size_in_pipe),
        ("user_categorical", user_categorical_pipe),
    ]
)
# - categorical
#     - gender
#     - climbing_<style>
#     - foot shape

# item features:
# - numerical
#    - size_in (stored in user_item)
# - categorical
#     - gender_id
#     - type

SKU_NUMERICAL_COLUMNS = ["size_in"]

shoe_size_in_pipe = Pipeline(
    steps=[
        ("shoe_size_in", ColumnSelector(SKU_NUMERICAL_COLUMNS)),
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

ITEM_TYPE_COLUMNS = ["item_type"]

item_type_pipe = Pipeline(
    steps=[
        ("column_selector", ColumnSelector(ITEM_TYPE_COLUMNS)),
        ("one_hot", OneHotEncoder(drop="first", sparse_output=False)),
    ]
)

sku_features_pipe = FeatureUnion(
    [
        ("shoe_size", shoe_size_in_pipe),
        ("climbing_year_in_style", climbing_years_in_style_pipe),
        ("item_type", item_type_pipe),
    ]
)

USED_COLUMNS = (
    EMBEDDING_COLUMNS
    + USER_CATEGORICAL_COLUMNS
    + SKU_NUMERICAL_COLUMNS
    + [TARGET_COLUMN]
)

target_pipe = LabelBinarizer()
