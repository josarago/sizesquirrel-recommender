from sklearn.preprocessing import (
    OrdinalEncoder,
    FunctionTransformer,
    OneHotEncoder,
    LabelBinarizer,
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

USER_CATEGORICAL_FEATURES = ["user_gender", "user_foot_shape"]

user_categorical_pipe = Pipeline(
    steps=[
        ("user_categories", ColumnSelector(USER_CATEGORICAL_FEATURES)),
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

shoe_size_in_pipe = Pipeline(
    steps=[
        ("shoe_size_in", ColumnSelector("size_in")),
        ("imputer", SimpleImputer(fill_value=-SIZE_IN_SCALE)),
        ("scale", FunctionTransformer(lambda x: x / SIZE_IN_SCALE)),
    ]
)


target_pipe = LabelBinarizer()
