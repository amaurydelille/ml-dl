import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

class Transformer():
    def __init__(self) -> None:
        pass

    def __drop(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = df.columns[df.isna().mean() > 0.5]
        dropped_df = df.drop(columns=columns_to_drop, axis=1)
        return dropped_df

    def __impute(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns

        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputed_numerical = imputer.fit_transform(df[numeric_cols])
        imputed_numerical_df = pd.DataFrame(
            data=imputed_numerical,
            columns=numeric_cols,
            index=df.index
        )

        imputed_df = pd.concat([imputed_numerical_df, df[categorical_cols]], axis=1)
        return imputed_df

    
    def __encode(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        
        if not categorical_columns:
            return df
        
        non_categorical_df = df.drop(columns=categorical_columns, axis=1)
        categorical_df = df[categorical_columns]
    
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_categorical = encoder.fit_transform(categorical_df)
        encoded_categorical_df = pd.DataFrame(
            data=encoded_categorical, 
            columns=encoder.get_feature_names_out(categorical_columns), 
            index=df.index
        )
        encoded_df = pd.concat([non_categorical_df, encoded_categorical_df], axis=1)
        return encoded_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        dropped_df = self.__drop(df)
        imputed_df = self.__impute(dropped_df)
        encoded_df = self.__encode(imputed_df)
        return encoded_df
        
        