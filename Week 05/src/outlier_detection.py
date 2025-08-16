import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class OutlierDetectionStrategy(ABC):

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        logging.info("Detecting outliers using IQR method")
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers[column] = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)
        return outliers

class OutlierDetector:
    def __init__(self, strategy):
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        logging.info("Starting outlier detection")
        outliers = self.strategy.detect_outliers(df, columns)
        logging.info("Outlier detection completed")
        return outliers

    def handle_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        logging.info("Handling outliers")
        outliers = self.detect_outliers(df, columns)
        outlier_count = outliers.sum(axis=1)
        rows_to_remove = outlier_count >= 2
        return df[~rows_to_remove]