import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data
from sklearn.utils.estimator_checks import check_estimator
from sklearn.exceptions import NotFittedError


class RegdateMonthsEncoder(TransformerMixin, BaseEstimator):
    """
    Learns minimum date during training, i.e. `fit()`, and computes number of 
    months since minimum date during `transform()` for both training and test data.

    Note that while this extends scikit-learn's transformer API, it is not fully compliant with `check_estimator()`.
    This transformer only handles datetime-parseable data, and will complain if any non-datetime-parseable data is provided.
    """

    def __init__(self, minimum_date=None):
        # Transformer can take in a minimum date, which it will use to compute the number of months.
        self.minimum_date = minimum_date

    def fit(self, X, y=None):
        # validate number of features and dtype
        X = validate_data(self, X, dtype='datetime64[ns]')

        # Compute minimum_date first
        if self.minimum_date is None:
            self.minimum_date_ = np.min(X, axis=0)
        else:
            self.minimum_date_ = np.full(X.shape[1], self.minimum_date)
        return self

    def transform(self, X, y=None):
        # validate number of features and dtype
        X = validate_data(self, X, reset=False, dtype='datetime64[ns]')

        # compute months since epoch
        min_months = self.minimum_date_.astype('datetime64[M]').astype(int)
        X_months = X.astype('datetime64[M]').astype(int)
        # find difference
        result = X_months - min_months
        return result

    def get_feature_names_out(self, input_features=None):
        if not input_features or len(input_features) == 1:
            return np.array(['regdate_months'])
        else:
            return np.array([f'regdate_months_{feature}' for feature in input_features])

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = [int] # does not preserve dtype
        return tags

class CategoricalInteractionEncoder(TransformerMixin, BaseEstimator):
    """
    Computes the one-hot encoding for 2 categorical variables and their interaction.
    Always encodes the first two columns in the provided array-like
    """

    def __init__(self, drop="first", handle_unknown="ignore"):
        self.drop = drop
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True, dtype=None) # preserve dtype, convert to numpy ndarray

        # create 2 one-hot encoders for the two categorical variables to be encoded
        self.ohe1_ = OneHotEncoder(drop=self.drop, handle_unknown=self.handle_unknown, sparse_output=False).fit(X[:, [0]])
        self.ohe2_ = OneHotEncoder(drop=self.drop, handle_unknown=self.handle_unknown, sparse_output=False).fit(X[:, [1]])

        # compute interaction from training set, check for columns that are fully zero and mask them out
        # (i.e. interactions that can theoretically exist from the set of categorical values, but don't actually exist in the training set)
        A = self.ohe1_.transform(X[:, [0]])
        B = self.ohe2_.transform(X[:, [1]])
        I = (A[:, :, None] * B[:, None, :]).reshape(A.shape[0], -1)

        self.names1_ = self.ohe1_.get_feature_names_out([self.feature_names_in_[0]])
        self.names2_ = self.ohe2_.get_feature_names_out([self.feature_names_in_[1]])
        self.interaction_mask_ = ~np.all(I == 0, axis=0)
        self.interaction_names_ = np.array([f"{a}*{b}" for a in self.names1_ for b in self.names2_])[self.interaction_mask_]

        return self

    def transform(self, X, y=None):
        X = validate_data(self, X, reset=False, dtype=None)

        A = self.ohe1_.transform(X[:, [0]])
        B = self.ohe2_.transform(X[:, [1]])
        I = (A[:, :, None] * B[:, None, :]).reshape(A.shape[0], -1)[:, self.interaction_mask_]

        return np.hstack([A, B, I])

    def get_feature_names_out(self, input_features=None):
        return np.concat([self.names1_, self.names2_, self.interaction_names_])


class LinearRegressionModel:
    """
    Model Specification
    - target: log(resale_price)
    - features:
        - regdate_months                : number of months since a specified start date)
        - flat type                     : flat type, e.g. "4 room", "5 room", "multi-generation"
        - town                          : town of the flat, e.g. "Bishan", "Toa Payoh", "Queenstown"
        - log(floor_area_sqm)           : logarithm of floor area in square metres
        - interaction (flat type, town) : interaction between flat type and town
    """

    def __init__(self):
        # initialize feature engineering pipeline
        self.ct = ColumnTransformer([
            ("regdate_months", RegdateMonthsEncoder(), ["date"]),
            ("ohe town flat_type", CategoricalInteractionEncoder(drop="first", handle_unknown="ignore"), ["town", "flat_type"]),
            ("log_floor_area_sqm", 
             FunctionTransformer(func=np.log, feature_names_out=lambda transformer, input_features: [f"{s} (log)" for s in input_features]),
             ["floor_area_sqm"]),
        ], remainder="drop")
        # initialize inference pipeline
        self.pipeline = Pipeline([
            ("features", self.ct),
            ("model", LinearRegression(fit_intercept=True)),
        ])

    ### Training Setup ###

    def import_data_and_train_model(self):
        # TODO: replace with API call to data.gov
        resale_data_path = Path() / "data" / "resale_flat_prices_20260720.csv" # update path as necessary
        ResalePrices = pd.read_csv(resale_data_path)

        # prepare training data and create model
        ResalePrices = self.preprocess_data(ResalePrices)
        y_train = self.get_target(ResalePrices)
        X_train = ResalePrices[['date', 'town', 'flat_type', 'floor_area_sqm']] # raw data, features are automatically computed by the pipeline
        self.pipeline.fit(X_train, y_train)

    def predict(self, df):
        return self.pipeline.predict(df)

    ### Data Wrangling ###

    # remove duplicates, outliers, transform strings into appropriate integer/date values etc.
    def preprocess_data(self, df):
        df = self.clean_data(df)
        df = self.coerce_month_to_datetime(df)
        df = self.coerce_remaining_lease_to_number_of_months(df)
        df = df.sort_values(by="date").reset_index(drop=True)
        return df
    
    def clean_data(self, df):
        """Clean bad values and drop duplicates"""

        # check for values that are not possible and drop them, e.g. having resale_price or floor_area_sqm <= 0
        df = df[df['resale_price'] > 0]
        df = df[df['floor_area_sqm'] > 0]
        # clean duplicates
        df = df.drop_duplicates(keep='first')
        # note: no outliers are removed for this model's training
        return df

    def coerce_month_to_datetime(self, df):
        """Transform month (str) -> date (datetime)"""

        df['date'] = pd.to_datetime(df['month'], format="%Y-%m", errors="coerce")
        # drop any rows with an invalid date
        df = df[~df['date'].isna()]
        return df

    def coerce_remaining_lease_to_number_of_months(self, df):
        """Transform remaining_lease (str) -> remaining_lease_months (int: number of months left in remaining lease)"""
        ym = df['remaining_lease'].str.extract(r"(?:(\d+) years?)?\s?(?:(\d+) months?)?").fillna(0)
        df['remaining_lease_months'] = ym.loc[:, 0].astype(int) * 12 + ym.loc[:, 1].astype(int)
        return df


    ### Feature Engineering ###

    def get_target(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Expected DataFrame input")
        if 'resale_price' not in df.columns:
            raise ValueError("Expected column 'resale_price' in DataFrame")

        return df['resale_price']

    def fit(self, train, y):
        # fits the feature engineering and prediction pipeline to training data
        self.pipeline.fit(X=train, y=y)
        return self

    def get_features(self, df):
        """
        Helper method for seeing the features the model has engineered
        Note: model must be fitted beforehand to use this method to get features.
        """
        data = self.ct.transform(df)
        return pd.DataFrame(data, columns=self.ct.get_feature_names_out())