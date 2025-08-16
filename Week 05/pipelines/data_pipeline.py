import os
import sys
import pandas as pd
from typing import Dict
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from outlier_detection import IQROutlierDetection
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStratergy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStratergy
from data_spiltter import SimpleTrainTestSplitStratergy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config

def data_pipeline(
        data_path: str='data/raw/ChurnModelling.csv',
        target_column: str='Exited',
        test_size: float=0.2,
        force_rebuild: bool=False,
    ):
    data_paths = get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    scaling_config = get_scaling_config()
    splitting_config = get_splitting_config()

    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])
    X_train_path = os.path.join(artifacts_dir, 'X_train.csv')
    X_test_path = os.path.join(artifacts_dir, 'X_test.csv')
    y_train_path = os.path.join(artifacts_dir, 'y_train.csv')
    y_test_path = os.path.join(artifacts_dir, 'y_test.csv')

    if (os.path.exists(X_train_path) and
        os.path.exists(X_test_path) and
        os.path.exists(y_train_path) and
        os.path.exists(y_test_path)):

        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path)
        y_test = pd.read_csv(y_test_path)

    ingestor = DataIngestorCSV()
    df = ingestor.ingest(data_path)
    print(f"Data Ingested: {df.shape[0]} rows, {df.shape[1]} columns")

    drop_handler = DropMissingValuesStrategy(columns['critical_columns'])
    age_handler = FillMissingValuesStrategy(
        method = 'mean',
        relevant_column = 'Age',
    )
    gender_handler = FillMissingValuesStrategy(
        relevant_column = 'Gender',
        is_custom_imputer = True,
        custom_imputer=GenderImputer(),
    )

    df = drop_handler.handle(df)
    df = age_handler.handle(df)
    df = gender_handler.handle(df)
    print(f"Data shape after imputation: {df.shape}")


data_pipeline()