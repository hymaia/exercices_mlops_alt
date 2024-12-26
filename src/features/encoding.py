import pandas as pd


class HolidaysEncoder:

    def __init__(self):
        pass

    @staticmethod
    def run(df):
        df['IsHoliday'] = df['IsHoliday'].apply(lambda x: 1 if x == True else 0)
        return df


class TypeEncoder:

    def __init__(self):
        pass

    @staticmethod
    def run(df):
        df['Type'] = df['Type'].map({'A': 0, 'B': 1, 'C': 2})
        return df


class DeptTargetEncoder:
    def __init__(self):
        self.dept_target_mean = None

    def fit(self, x_train, y_train):
        df_join_x_and_y = pd.concat([x_train, y_train], axis=1)
        self.dept_target_mean = df_join_x_and_y.groupby('Dept')['Weekly_Sales'].mean().to_dict()
        return self

    def transform(self, df):
        df['Dept'] = df['Dept'].map(self.dept_target_mean)
        return df


class StoreTargetEncoder:
    def __init__(self):
        self.store_target_mean = None

    def fit(self, x_train, y_train):
        df_join_x_and_y = pd.concat([x_train, y_train], axis=1)
        self.store_target_mean = df_join_x_and_y.groupby('Store')['Weekly_Sales'].mean().to_dict()
        return self

    def transform(self, df):
        df['Store'] = df['Store'].map(self.store_target_mean)
        return df
