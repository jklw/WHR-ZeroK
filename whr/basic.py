from whr.imports import *

eloPerNaturalRating : float = 400 / math.log(10)
naturalRatingPerElo : float = 1 / eloPerNaturalRating
naturalRatingPerEloSquared : float = naturalRatingPerElo**2

def naturalRatingToElo(r): return r * eloPerNaturalRating + 1500

def eloToNaturalRating(e): return (e - 1500) * naturalRatingPerElo

def zkNaturalRatingVariancePerDay(totalWeight):
    return naturalRatingPerEloSquared * 200000 / (totalWeight + 400)

zkNaturalRatingVariancePerGame : float = naturalRatingPerEloSquared * 500


def slice2arr(sl : slice):
    return np.arange(sl.start, sl.stop)

def cacheFileName(name : str, hash : str) -> str: return f'cache/{name}_{hash}.pickle'

def hashArgs(args : list, kwargs : dict[str,Any]) -> str:
    hash = hashlib.md5()

    def hashOne(a):
        if(isinstance(a, (pd.Index, pd.DataFrame, pd.Series))):
            hash.update(b'pd')
            hash.update(pd.util.hash_pandas_object(a).to_numpy())
        else:
            hash.update(a)

    for a in args:
        hash.update(b'a')
        hashOne(a)

    for k, a in kwargs:
        hash.update(b'k')
        hashOne(k)
        hash.update(b'a')
        hashOne(a)

    return hash.hexdigest()

# _P = typing.ParamSpec('P')
T = typing.TypeVar('T')

def version(ver):
    """ Invalidates cached data when changed."""
    def decorator(func):
        func.version = ver
        return func
    return decorator

# def cached(name : str, create : typing.Callable[_P, typing._T], *args : _P.args, **kwargs : _P.kwargs) -> typing._T: 
def cached(name : str, func : typing.Callable[..., T], *args : list, **kwargs : dict[str,Any]) -> T: 
    if not hasattr(func,'version'):
        raise ValueError(f'function {func} needs a version decorator/attribute')

    ver = func.version
    fn = cacheFileName(name, hashArgs(args, kwargs))
    try:
        p = pd.read_pickle(fn)
    except FileNotFoundError:
        p = None
        
    if p is not None:
        pVer = p['version']
        if pVer  == ver:
            # print(f'using cached data from {fn}')
            return p['data']
        else:
            print(f'Recreating because cached version = {pVer} != {ver}')
    
    data = func(*args, **kwargs)
    pd.to_pickle({'version':ver, 'data':data}, fn)

    return data

