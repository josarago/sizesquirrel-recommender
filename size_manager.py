import os
import pandas as pd

"""
Workflow:
- load the training set
- get all items from training set (brand + model + gender)
- get size map from file `data/sizing_systems/<brand_name>/<model> - <year>
- convert all sizes to european (make sure men US and women US map to a single European sizing scale)
- create heuristic for data augmentation:
	- for instance, if rating for a given size ~ 
"""


BRANDS = ("la sportiva",)
SIZING_SYSTEMS_DIR = os.path.join("data", "sizing_systems")


class SizeManager:
    sizing_systems_dir = SIZING_SYSTEMS_DIR

    def __init__(self, df, brand_name="la sportiva", model="tc pro", gender="men"):
        self.df = df
        self._brand_name = brand_name
        self._model = model
        self._gender = gender

    @property
    def brand_name(self):
        return self._brand_name

    @property
    def shoe_model(self):
        return self._shoe_model

    @property
    def gender(self):
        return self._gender

    def get_known_sizes(self):
        return sorted(
            self.df.query(f"brand_name=='{self.brand_name}'")
            .query(f"model=='{self.model}'")
            .query(f"gender=='{self.gender}'")["size"]
            .unique()
        )

    def create_all_skus(
        self, known_sizes, brand_name="la sportiva", model="tc pro", gender="men"
    ):
        _df = pd.DataFrame(data={"size": known_sizes})
        _df["brand_name"] = brand_name
        _df["model"] = model
        _df["gender"] = gender
        _df["sku_id"] = trainer.compute_sku_id(_df)
        return _df.sort_values("size")
