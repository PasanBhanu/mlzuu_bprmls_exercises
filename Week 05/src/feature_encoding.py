import logging
import pandas as pd
import os
import json
from enum import Enum
from typing import Dict, List
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class FeatureEncodingStrategy(ABC):

    @abstractmethod
    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        pass


class VariableType(str, Enum):
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'


class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nominal_columns):
        self.nominal_columns = nominal_columns
        self.encoder_dicts = {}
        os.makedirs('artifacts/encode', exist_ok=True)

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.nominal_columns:
            unique_values = df[column].unique()
            encoding_dict = {value: idx for idx, value in enumerate(unique_values)}
            self.encoder_dicts[column] = encoding_dict

            encoder_path = os.path.join('artifacts/encode', column + '_encoder.json')
            with open(encoder_path, 'w') as f:
                json.dump(self.encoder_dicts, f)

            df[column] = df[column].map(encoding_dict)

        return df

    def get_encoder_dicts(self) -> Dict[str, Dict[str, int]]:
        return self.encoder_dicts


class OrdinalEncodingStratergy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings):
        self.ordinal_mappings = ordinal_mappings

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for column, mapping in self.ordinal_mappings.items():
            df[column] = df[column].map(mapping)

        return df
