from IPython.display import HTML
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from dataclasses import field
from numpy.random import Generator
from pandas import Series, DataFrame
from typing import Any, Optional, Tuple, Union, TYPE_CHECKING, cast
import pydantic
import arviz as az
import gzip
import hashlib
import heapq
import itertools as iter
import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandera as pa
import pandera.typing as pat
from pandera import DataFrameModel
import pymc as pm
import pytensor as pt0
import pytensor.d3viz as d3v
import pytensor.sparse as pts
import pytensor.tensor as pt
import scipy.sparse as scs
import typing
import xarray as xa
import re

if TYPE_CHECKING:
    import _typeshed