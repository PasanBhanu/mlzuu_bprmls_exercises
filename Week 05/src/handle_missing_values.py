import groq
import logging
import pandas as pd
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()


class MissingValueHandlingStrategy(ABC):

    @abstractmethod
    def handle(self, df: pd.DataFrame) ->pd.DataFrame:
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, critical_columns=[]):
        self.critical_columns = critical_columns
        logging.info(f"Dropping rows with missing values in critical columns: {critical_columns}")

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleared = df.dropna(subset=self.critical_columns)
        n_dropped = len(df) - len(df_cleared)
        logging.info(f"Dropped {n_dropped} rows")
        return df_cleared

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

class GenderPrediction(BaseModel):
    firstname: str
    lastname: str
    pred_gender: Gender

class GenderImputer:
    def __init__(self):
        self.groq_client = groq.Groq()

    def _predict_gender(self, firstname: str, lastname: str):
        prompt = f"""What is the most likely gender (Male or Female) for someone with the firstname{firstname} and lastname {lastname}?

        Your response only consists of one word: Male or Female"""

        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        predicted_gender = response.choices[0].message.content.strip()
        prediction = GenderPrediction(firstname=firstname, lastname=lastname, pred_gender=predicted_gender)
        return prediction.pred_gender

    def impute(self, df):
        missing_gender_index = df['Gender'].isnull()
        for idx in df[missing_gender_index].index:
            firstname = df.loc[idx, 'Firstname']
            lastname = df.loc[idx, 'Lastname']
            gender = self._predict_gender(firstname, lastname)
            if gender:
                df.loc[idx, 'Gender'] = gender
            else:
                print(f"{firstname} {lastname} : No Gender Detected")

class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """
    Missing -> Mean (Age)
            -> Custom (Gender)
    """

    def __init__(
            self,
            method='mean',
            fill_value=None,
            relevant_column=None,
            is_custom_imputer=False,
            custom_imputer=None
    ):
        self.method = method
        self.fill_value = fill_value
        self.relevant_column = relevant_column
        self.is_custom_imputer = is_custom_imputer
        self.custom_imputer = custom_imputer

    def handle(self, df):
        if self.is_custom_imputer:
            return self.custom_imputer.impute(df)

        df[self.relevant_column] = df[self.relevant_column].fillna(df[self.relevant_column].mean())
        logging.info(f"Filling missing values in {df.shape[0]} rows")
        return df