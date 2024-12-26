import pandas as pd


class DataCollector:

    def __init__(self):
        pass

    @staticmethod
    def merge_sales_and_features(df_sales: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
        return df_sales.merge(df_features, on=["Store", "Date", "IsHoliday"])

    @staticmethod
    def merge_sales_and_stores(df_sales: pd.DataFrame, df_stores: pd.DataFrame) -> pd.DataFrame:
        return df_sales.merge(df_stores, on=["Store"])

    def gather_data(self, path_dataset_sales, path_features, path_store) -> tuple:
        """
        Reads sales dataset (train or test) and merges them with features and stores datasets.
        :return:
        """
        # read data
        df_sales = pd.read_csv(path_dataset_sales)
        df_features = pd.read_csv(path_features)
        df_stores = pd.read_csv(path_store)

        # merge together
        df_sales = self.merge_sales_and_features(df_sales, df_features)
        df_sales = self.merge_sales_and_stores(df_sales, df_stores)

        return df_sales
