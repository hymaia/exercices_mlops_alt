import pandas as pd


class HolidaysComputer:
    def __init__(self):
        pass

    @staticmethod
    def extract_dates(df: pd.DataFrame) -> pd.DataFrame:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Day"] = df["Date"].dt.day.astype(int)
        df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
        df["Month"] = df["Date"].dt.month.astype(int)
        df["Year"] = df["Date"].dt.year.astype(int)
        return df

    @staticmethod
    def compute_days_until_christmas(df: pd.DataFrame) -> pd.DataFrame:
        serie_current_date = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        serie_current_christmas = pd.to_datetime(
            df["Year"].astype(str) + "-12-24", format="%Y-%m-%d"
        )
        df["Days_to_Christmas"] = (
            serie_current_christmas - serie_current_date
        ).dt.days.astype(int)
        return df

    @staticmethod
    def compute_days_until_thanksgiving(df: pd.DataFrame) -> pd.DataFrame:
        serie_current_date = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        serie_current_christmas = pd.to_datetime(
            df["Year"].astype(str) + "-11-24", format="%Y-%m-%d"
        )
        df["Days_to_Thansksgiving"] = (
            serie_current_christmas - serie_current_date
        ).dt.days.astype(int)
        return df

    @staticmethod
    def compute_superbowl_week(df: pd.DataFrame) -> pd.DataFrame:
        df["SuperBowlWeek"] = df["Week"].apply(lambda x: 1 if x == 6 else 0)
        return df

    @staticmethod
    def compute_labor_day_week(df: pd.DataFrame) -> pd.DataFrame:
        df["LaborDayWeek"] = df["Week"].apply(lambda x: 1 if x == 36 else 0)
        return df

    @staticmethod
    def compute_thanksgiving_week(df: pd.DataFrame) -> pd.DataFrame:
        df["ThanksgivingWeek"] = df["Week"].apply(lambda x: 1 if x == 47 else 0)
        return df

    @staticmethod
    def compute_christmas_week(df: pd.DataFrame) -> pd.DataFrame:
        df["ChristmasWeek"] = df["Week"].apply(lambda x: 1 if x == 52 else 0)
        return df

    def run(self, df_to_transform: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts dates and computes specials days and weeks in the US calendar.
        :param df_to_transform: (Dataframe) containing input features
        :return: df_to_transform + new features
        """
        # extract dates and computes days before Christmas and thanksgiving
        df_to_transform = self.extract_dates(df_to_transform)
        df_to_transform = self.compute_days_until_thanksgiving(df_to_transform)
        df_to_transform = self.compute_days_until_christmas(df_to_transform)

        # special dates in the US
        df_to_transform = self.compute_superbowl_week(df_to_transform)
        df_to_transform = self.compute_labor_day_week(df_to_transform)
        df_to_transform = self.compute_thanksgiving_week(df_to_transform)
        df_to_transform = self.compute_christmas_week(df_to_transform)

        return df_to_transform
