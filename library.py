from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import types
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices

# Custom Imports to make multiple files work together
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# My custom library
from customDropColumnsTransformer import CustomDropColumnsTransformer
from customMapingTransformer import CustomMappingTransformer
from customOHETransformer import CustomOHETransformer