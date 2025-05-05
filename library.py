from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import types
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable

from sklearn.preprocessing import _encoders
# from feature_engine.encoding import MeanEncoder
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier



import sklearn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split



import warnings


sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices

# State Variables (Chapter 7)
titanic_variance_based_split = 107
customer_variance_based_split = 113 

# Library Transformers
class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

    #your code below

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit method (does nothing — required for scikit-learn compatibility).
        """

        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by dropping or keeping columns.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with columns dropped or kept.
        """
        # Defensive Copy
        X_ = X.copy()

        missing_cols = [col for col in self.column_list if col not in X_.columns]

        if self.action == 'keep':
            # Error if trying to keep missing columns
            assert not missing_cols, f'CustomDropColumnsTransformer.transform unknown columns to keep: {missing_cols}'
            return X_[self.column_list]

        elif self.action == 'drop':
            # Warning if dropping missing columns
            if missing_cols:
                warnings.warn(f'DropColumnsTransformer does not contain these columns to drop: {missing_cols}' , UserWarning)
            return X.drop(columns=self.column_list, errors='ignore')

        else:
            raise ValueError(f'Unsupported action: {self.action}. Must be "drop" or "keep".')
        
class CustomOHETransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that performs one-hot encoding on a specified column.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies one-hot encoding to a specified column using
    pandas' `pd.get_dummies` function.
    """
    def __init__(self, target_column: str) -> None:
        """
        Initialize the CustomOHETransformer.

        Parameters
        ----------
        target_column : str
            The name of the column to one-hot encode.
        """
        self.target_column = target_column
        assert isinstance(target_column, str), f'{self.__class__.__name__} constructor expected string but got {type(target_column)} instead.'

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with one-hot encoding applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if target_column is not in X.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns

        X_ = X.copy()
        return pd.get_dummies(X_, columns=[self.target_column], dtype=int) # Specify int to avoid "True" or "False"

class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
        category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result

class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """

    def __init__(self, target_column):
        self.target_column = target_column
        self.high_wall = None
        self.low_wall = None

        assert isinstance(target_column, str), f'expected str but got {type(target_column)} instead.'

    def fit(self, X, y=None):
        """
        Compute the high and low walls for 3-sigma clipping.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        assert pd.api.types.is_numeric_dtype(X[self.target_column]), f'expected int or float in column {self.target_column}'

        # Calculate mean and std
        mean = X[self.target_column].mean()
        std = X[self.target_column].std()

        # Compute the high and low walls
        self.high_wall = mean + 3 * std
        self.low_wall = mean - 3 * std

    def transform(self, X, y=None):
        """
        Clip values in the target column to be within 3-sigma of the mean.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        assert pd.api.types.is_numeric_dtype(X[self.target_column]), f'expected int or float in column {self.target_column}'
        assert self.high_wall is not None and self.low_wall is not None, f'Sigma3Transformer.fit has not been called.'

        # Clip acording to walls
        X[self.target_column] = X[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)

        # Reset index
        X = X.reset_index(drop=True)

        return X

    def fit_transform(self, X, y=None):
        """
        Compute the high and low walls for 3-sigma clipping and clip values in the target column.
        """
        self.fit(X, y)
        return self.transform(X, y)

class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """

    def __init__(self, target_column= None, fence='outer'):
        # Parameters
        self.target_column = target_column
        self.fence = fence
        
        # Attributes
        self.inner_low = None
        self.outer_low = None
        self.inner_high = None
        self.outer_high = None

        # Validation
        assert isinstance(target_column, str), f'expected str but got {type(target_column)} instead.'
        assert fence in ['inner', 'outer'], f'unknown fence {fence}'

    def fit(self, X, y=None):
        """
        Compute the inner and outer walls for Tukey's fences.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'

        # Calculate quartiles and IQR
        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        iqr = q3 - q1

        # print(f'Fence: {self.fence}')

        if self.fence == 'inner':
            self.inner_low = q1 - 1.5 * iqr
            self.inner_high = q3 + 1.5 * iqr

        elif self.fence == 'outer':
            self.outer_low = q1 - 3.0 * iqr
            self.outer_high = q3 + 3.0 * iqr            
        else:
            raise ValueError(f'Unknown fence type: {self.fence}')

        return self

    def transform(self, X, y=None):
        """
        Clip values in the target column based on Tukey's fences.
        """
    
        assert isinstance(X, pd.DataFrame), f'expected DataFrame but got {type(X)}'
        assert self.target_column in X.columns, f'unknown column {self.target_column}'
        

        if self.fence == 'inner':
            low_bound, high_bound = self.inner_low, self.inner_high
        elif self.fence == 'outer':
            low_bound, high_bound = self.outer_low, self.outer_high
        else:
            raise ValueError(f'Unknown fence type: {self.fence}')
        
        assert low_bound is not None and high_bound is not None, 'fit() must be called before transform().'
        
        # Do transform and return with fixed indicies
        X[self.target_column] = X[self.target_column].clip(lower=low_bound, upper=high_bound)
        return X.reset_index(drop=True)

        

    def fit_transform(self, X, y=None):
        """
        Compute the inner and outer walls for Tukey's fences and clip values in the target column.
        """
        X_copy = X.copy()
        self.fit(X_copy, y)
        return self.transform(X_copy, y)

class CustomRobustTransformer(BaseEstimator, TransformerMixin):
    """
    Applies robust scaling to a specified column in a pandas DataFrame.

    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method. Handles binary columns
    gracefully by skipping scaling for them.

    Parameters
    ----------
    column : str
        The name of the column to be scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : float or None
        The interquartile range of the target column. None if the column is binary.
    med : float or None
        The median of the target column. None if the column is binary.
    bin : bool
        Flag indicating if the column is binary.
    """
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.iqr = None
        self.med = None
        self.bin = None

    def fit(self, X: pd.DataFrame, y=None) -> 'CustomRobustTransformer':
        """
        Computes the IQR and median of the target column.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : CustomRobustTransformer
            The fitted transformer.
        """
        if self.target_column not in X.columns:
            raise ValueError(f"CustomRobustTransformer.fit: Unrecognized column '{self.target_column}'")

        # Check if the column is binary
        if X[self.target_column].nunique() <= 2 and X[self.target_column].isin([0, 1]).all():
            self.bin = True
            return self
        else:
            self.iqr = X[self.target_column].quantile(0.75) - X[self.target_column].quantile(0.25)
            self.med = X[self.target_column].median()
            return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Applies robust scaling to the target column.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the scaled column.
        """
        if self.bin:
            return X

        if self.iqr is None or self.med is None:
            raise AssertionError("This CustomRobustTransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        if self.iqr < 1e-10:
            raise ValueError(f"CustomRobustTransformer.transform: IQR is too small for column '{self.target_column}'")
        

        X[self.target_column] = (X[self.target_column] - self.med) / self.iqr
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fits the transformer and applies robust scaling to the target column.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the scaled column.
        """
        self.fit(X)
        return self.transform(X)

class CustomKNNTransformer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using KNN.

    This transformer wraps the KNNImputer from scikit-learn and hard-codes
    add_indicator to be False. It also ensures that the input and output
    are pandas DataFrames.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction. Possible values:
        "uniform" : uniform weights. All points in each neighborhood
        are weighted equally.
        "distance" : weight points by the inverse of their distance.
        In this case, closer neighbors of a query point will have a
        greater influence than neighbors which are further away.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.knn_imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            add_indicator=False
        )

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        """
        Fit the KNN imputer on the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to fit the imputer on.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : CustomKNNTransformer
            The fitted transformer.
        """
        self.knn_imputer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by imputing missing values.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        if not hasattr(self, 'knn_imputer'):
            raise ValueError(
                "This CustomKNNTransformer instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        X_imputed = self.knn_imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the KNN imputer on the input DataFrame and transform it.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to fit and transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        return self.fit(X, y).transform(X)

class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    """
    A target encoder that applies smoothing and returns np.nan for unseen categories.

    Parameters:
    -----------
    col: name of column to encode.
    smoothing : float, default=10.0
        Smoothing factor. Higher values give more weight to the global mean.
    """

    def __init__(self, col: str, smoothing: float =10.0):
        self.col = col
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_dict_ = None

    def fit(self, X, y):
        """
        Fit the target encoder using training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead.'
        assert self.col in X, f'{self.__class__.__name__}.fit column not in X: {self.col}. Actual columns: {X.columns}'
        assert isinstance(y, Iterable), f'{self.__class__.__name__}.fit expected Iterable but got {type(y)} instead.'
        assert len(X) == len(y), f'{self.__class__.__name__}.fit X and y must be same length but got {len(X)} and {len(y)} instead.'

        #Create new df with just col and target - enables use of pandas methods below
        X_ = X[[self.col]]
        target = self.col+'_target_'
        X_[target] = y

        # Calculate global mean
        self.global_mean_ = X_[target].mean()

        # Get counts and means
        counts = X_[self.col].value_counts().to_dict()    #dictionary of unique values in the column col and their counts
        means = X_[target].groupby(X_[self.col]).mean().to_dict() #dictionary of unique values in the column col and their means

        # Calculate smoothed means
        smoothed_means = {}
        for category in counts.keys():
            n = counts[category]
            category_mean = means[category]
            # Apply smoothing formula: (n * cat_mean + m * global_mean) / (n + m)
            smoothed_mean = (n * category_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
            smoothed_means[category] = smoothed_mean

        self.encoding_dict_ = smoothed_means

        return self

    def transform(self, X):
        """
        Transform the data using the fitted target encoder.
        Unseen categories will be encoded as np.nan.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        """

        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.encoding_dict_, f'{self.__class__.__name__}.transform not fitted'

        X_ = X.copy()

        # Map categories to smoothed means, naturally producing np.nan for unseen categories, i.e.,
        # when map tries to look up a value in the dictionary and doesn't find the key, it automatically returns np.nan. That is what we want.
        X_[self.col] = X_[self.col].map(self.encoding_dict_)

        return X_

    def fit_transform(self, X, y):
        """
        Fit the target encoder and transform the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        return self.fit(X, y).transform(X)

# Custom Function
def find_random_state(
    features_df: pd.DataFrame,
    labels: Iterable,
    transformer: TransformerMixin,
    n: int = 200
    ) -> Tuple[int, List[float]]:
    """
    Finds an optimal random state for train-test splitting based on F1-score stability.

    This function iterates through `n` different random states when splitting the data,
    applies a transformation pipeline, and trains a K-Nearest Neighbors classifier.
    It calculates the ratio of test F1-score to train F1-score and selects the random
    state where this ratio is closest to the mean.

    Parameters
    ----------
    features_df : pd.DataFrame
        The feature dataset.
    labels : Union[pd.Series, List]
        The corresponding labels for classification (can be a pandas Series or a Python list).
    transformer : TransformerMixin
        A scikit-learn compatible transformer for preprocessing.
    n : int, default=200
        The number of random states to evaluate.

    Returns
    -------
    rs_value : int
        The optimal random state where the F1-score ratio is closest to the mean.
    Var : List[float]
        A list containing the F1-score ratios for each evaluated random state.

    Notes
    -----
    - If the train F1-score is below 0.1, that iteration is skipped.
    - A higher F1-score ratio (closer to 1) indicates better train-test consistency.
    """

    model = KNeighborsClassifier(n_neighbors=5)
    Var: List[float] = []  # Collect test_f1/train_f1 ratios

    for i in range(n):
        train_X, test_X, train_y, test_y = train_test_split(
            features_df, labels, test_size=0.2, shuffle=True,
            random_state=i, stratify=labels  # Works with both lists and pd.Series
        )

        # Apply transformation pipeline
        transform_train_X = transformer.fit_transform(train_X, train_y)
        transform_test_X = transformer.transform(test_X)

        # Train model and make predictions
        model.fit(transform_train_X, train_y)
        train_pred = model.predict(transform_train_X)
        test_pred = model.predict(transform_test_X)

        train_f1 = f1_score(train_y, train_pred)

        if train_f1 < 0.1:
            continue  # Skip if train_f1 is too low

        test_f1 = f1_score(test_y, test_pred)
        f1_ratio = test_f1 / train_f1  # Ratio of test to train F1-score

        Var.append(f1_ratio)

    mean_f1_ratio: float = np.mean(Var)
    rs_value: int = np.abs(np.array(Var) - mean_f1_ratio).argmin()  # Index of value closest to mean

    return rs_value, Var

# Custom Pipelines
titanic_transformer = Pipeline(steps=[
    # From Chapter 7
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', CustomTargetTransformer(col='Joined', smoothing=10)),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer(target_column='Age')),
    ('scale_fare', CustomRobustTransformer(target_column='Fare')),
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)


customer_transformer = Pipeline(steps=[
    # From Chapter 7
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', CustomTargetTransformer(col='ISP')),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer(target_column='Age')), #from 5
    ('scale_time spent', CustomRobustTransformer(target_column='Time Spent')), #from 5
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)