import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocess(data):

    df = pd.DataFrame(data)

    numerical_features = ['feature_start', 'feature_end', 'feature_length']
    categorical_features = ['clinical_intervention'] # label. do we add this here?

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    # do we need the "not provided"? should probabl drop right"

    preprocessor = ColumnTransformer()

# to delete after
def load_dataframe(data):
    # transforms json array to dataframe
    return pd.DataFrame(data)


    