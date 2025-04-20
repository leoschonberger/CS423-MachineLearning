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

