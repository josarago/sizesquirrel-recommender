import pathlib
import pandas as pd

from config import SIZING_SYSTEM_DIR_PATH


class SizeManager:
    def __init__(
        self, user_item_df: pd.DataFrame, sizing_system_dir_path=SIZING_SYSTEM_DIR_PATH
    ) -> None:
        self.user_item_df = user_item_df
        self.sizing_system_dir_path = sizing_system_dir_path
        self.sizing_df = self.get_sizing_df()

    def get_sizing_df(
        self,
    ):
        # we store all .csv files in a list
        filelist = []
        for root, dirs, files in os.walk(self.sizing_system_dir_path):
            for file in files:
                # append the file name to the list
                if pathlib.Path(file).suffix == ".csv":
                    filelist.append(pathlib.PurePath(root, file))

        # for each file, load the data
        df = pd.DataFrame()
        for file_path in filelist:
            p = pathlib.Path(file_path)
            model = p.stem
            brand_name = p.parent.name
            _df = pd.read_csv(file_path, usecols=["us_men", "euro", "us_women"])
            _df["model"] = model
            _df["brand_name"] = brand_name
            df = pd.concat([df, _df])

        men_df = df[["brand_name", "model", "us_men", "euro"]].rename(
            columns={"us_men": "us_size", "euro": "euro_size"}
        )
        men_df.insert(2, "shoe_gender", "men")

        women_df = df[["brand_name", "model", "us_women", "euro"]].rename(
            columns={"us_women": "us_size", "euro": "euro_size"}
        )
        women_df.insert(2, "shoe_gender", "women")
        sizing_df = pd.concat([men_df, women_df]).dropna(subset=["us_size"])
        sizing_df["us_size"] = sizing_df["us_size"].apply(
            lambda x: x + ".0" if not ("." in x) else x
        )
        return sizing_df

    @staticmethod
    def convert_to_euro_size(row):
        if row["sizing_system"] == "EURO":
            return row["size"]
        if not (row["converted_euro_size"] is None):
            return row["converted_euro_size"]
        else:
            return pd.NaT

    def convert_to_euro_sizes(self):
        """
        user_item_df: user-item dataframe, typically .df attrubute of Trainer instance
        sizing_df:
        """
        merged_df = (
            self.user_item_df
            # [
            #     ["brand_name", "model", "shoe_gender", "size", "sizing_system"]
            # ]
            .astype({"size": str})
            .merge(
                self.sizing_df,
                left_on=["brand_name", "model", "shoe_gender", "size"],
                right_on=["brand_name", "model", "shoe_gender", "us_size"],
                how="left",
            )
            .rename(columns={"euro_size": "converted_euro_size"})
        )
        merged_df["euro_size"] = merged_df.apply(self.convert_to_euro_size, axis=1)
        return merged_df.drop(columns=["us_size"])


if __name__ == "__main__":
    size_manager = SizeManager(None)
    print(size_manager.sizing_df)
