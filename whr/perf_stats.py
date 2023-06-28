from whr.basic import *
from whr.preprocess import *

# `extendPerfStats` moved to `WHRModel`

def loadPerfStats(filePath='perfStats.pickle') -> DataFrame:
    rows = []

    with open(filePath,'rb') as f:
        try:
            while True:
                rows.append(pd.read_pickle(f))
        except EOFError:
            pass

    return pd.concat(rows, ignore_index=True)