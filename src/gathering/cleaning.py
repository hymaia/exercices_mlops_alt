import pandas as pd


class DataCleaner:
    def __init__(self):
        self.mean_cpi = None
        self.mean_unemployment = None

    def fit(self, df_train: pd.DataFrame):
        """
        Fits the average values of columns in order to deal with missing values.
        :param df_train: (DataFrame) used to fit the average values
        :return: (no return)
        """
        self.mean_cpi = df_train["CPI"].mean()
        self.mean_unemployment = df_train["Unemployment"].mean()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the columns with the mean values computed in the fit method.
        :param df: (DataFrame) to clean missing values
        :return: df (DataFrame) cleaned with no missing values
        """
        list_markdown_cols = ["MarkDown%s" % i for i in range(1, 6)]
        df[list_markdown_cols] = df[list_markdown_cols].fillna(0.0)
        df["CPI"].fillna(self.mean_cpi, inplace=True)
        df["Unemployment"].fillna(self.mean_unemployment, inplace=True)

        if df.isnull().sum().sum() > 0:
            raise ValueError("There are still missing values in the DataFrame.")

        return df
