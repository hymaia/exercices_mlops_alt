from features.encoding import DeptTargetEncoder, StoreTargetEncoder
from features.markdown import add_total_markdown
from features.holidays import HolidaysComputer
import pandas as pd


class FeaturesEngineering:
    def __init__(self):
        self.store_target_encoder = None
        self.dept_target_encoder = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the estimators on train dataset
        :return:
        """
        self.dept_target_encoder = DeptTargetEncoder().fit(x_train, y_train)
        self.store_target_encoder = StoreTargetEncoder().fit(x_train, y_train)
        return self

    def transform(self, df_to_transform: pd.DataFrame) -> pd.DataFrame:
        """
        Applies features engineering on a test dataset
        :return:
        """
        # markdown sum
        df_to_transform = add_total_markdown(df_to_transform)

        # holidays and special weeks
        df_to_transform = HolidaysComputer().run(df_to_transform)

        # encoding
        df_to_transform = self.dept_target_encoder.transform(df_to_transform)
        df_to_transform = self.store_target_encoder.transform(df_to_transform)

        return df_to_transform
