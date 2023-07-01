from whr.preprocess import *
from whr.perf_stats import *
import enum
import scipy.special

def makeConst(value, size):
    return pt.repeat(pt.constant(value), size)

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RatingVector:
    vector : pt.TensorVariable
    ixLookup : pat.Series[int]
    """ 
    Index:
        player : int (the playerid)
        day : datetime64
    Values:
        Index into the vector
    """

advantageClasses = np.array([
    '1v2', '2v3', '3v4', 
    '4v5', '5v6', '6v7', 
    '7v8', '8v9', '9v10', 
    '10v11', '11v12', '12v13',
    '13v14', '14v15', '15v16+'])
ratingclassCoords = np.array(['low rating', 'high rating'])

class ExtraPlayerAdvantageType(enum.Enum):
    NONE = 0,
    BY_TEAM_SIZE = 1,
    BY_TEAM_SIZE_AND_BPST = 2,
    DUMMY = 3, # Constrains the parameters to be almost 0; effectively like NONE but lets us do model comparison

ExtraPlayerAdvantageType.NONE.titlePart = 'unadjusted WHR model'
ExtraPlayerAdvantageType.BY_TEAM_SIZE.titlePart = 'simple adjusted model'
ExtraPlayerAdvantageType.BY_TEAM_SIZE_AND_BPST.titlePart = 'BPST-adjusted model'
ExtraPlayerAdvantageType.DUMMY.titlePart = 'unadjusted WHR model'

def ratingHighness(idata, rat):
    return scipy.special.expit( \
                    (rat - idata.constant_data['ratingClassCenter'].values[0]) \
                    * idata.constant_data['ratingClassSharpness'].values[0])

def eloHighness(idata, elo):
    return ratingHighness(idata, eloToNaturalRating(elo))


def makeExtraPlayerAdvantageVariables(da : PreprocessedData, ratings : RatingVector, gameLogits : pt.TensorVariable, typ : ExtraPlayerAdvantageType) -> pt.TensorVariable :
    """
    Must be called in `with model:` block.

    Returns: The adjusted gameLogits
    """

    if(typ==ExtraPlayerAdvantageType.NONE): return gameLogits
    elif(typ == ExtraPlayerAdvantageType.BY_TEAM_SIZE): isByBPST = False
    elif(typ == ExtraPlayerAdvantageType.BY_TEAM_SIZE_AND_BPST): isByBPST = True
    elif(typ == ExtraPlayerAdvantageType.DUMMY): isByBPST = False
    else: raise ValueError(typ)

    epadcPriorSigma = naturalRatingPerElo * 1000

    model : pm.Model = pm.modelcontext(None)
    model.add_coord('advc', advantageClasses)

    if isByBPST:
        model.add_coord('ratingclass', ratingclassCoords)
        ratingClassCenter = pm.ConstantData('ratingClassCenter', eloToNaturalRating(2000))
        ratingClassSharpness = pm.ConstantData('ratingClassSharpness', 1/(200*naturalRatingPerElo) )

    nAdvc = len(advantageClasses)

    if typ != ExtraPlayerAdvantageType.DUMMY:
        # Extra player advantage coefficient:
        epadc = pm.Normal('epadc', \
            sigma = [[epadcPriorSigma]*2] * nAdvc if isByBPST else [epadcPriorSigma]*nAdvc, \
            dims=('advc','ratingclass') if isByBPST else ('advc',))
    else:
        epadc = pm.Uniform('epadc', lower = [-naturalRatingPerElo] * nAdvc, upper = [naturalRatingPerElo] * nAdvc, dims = ('advc',))

    # ratingClassCenter = pm.Normal('ratingClassCenter', sigma=naturalRatingPerElo * 500)
    # ratingClassSharpness = 1/(100*naturalRatingPerElo) * pm.LogNormal('ratingClassSharpness', sigma=1)


    # epadc = pm.Mixture('epadc', 
    #         w = [0.5, 0.5],
    #         comp_dists = [
    #             pm.Normal.dist(sigma = [naturalRatingPerElo * 1] * len(advantageClasses)), 
    #             pm.Normal.dist(sigma = [epadcPriorSigma] * len(advantageClasses)), 
    #         ],
    #         dims=('advc',))

    games = da.games
    unevenGames = games.assign(gameIx = range(len(games)))[games.winnerCount != games.loserCount].assign( \
        # sign: 1 if the winner team is larger, -1 if the loser team is larger
        sign_largeTeamWin = lambda u: u.winnerCount - u.loserCount,
        smallTeamSize = lambda u: u.winnerCount.clip(upper=u.loserCount), # Elementwise minimum       
        # iAdvantageClass = lambda u: u.smallTeamSize.clip(upper=len(advantageClasses)) - 1
        # smallTeamPlayers = lambda u: u.winners.where(u.sign_largeTeamWin < 0, u.losers)
        )

    assert (unevenGames.sign_largeTeamWin.abs() == 1).all(), 'Games with player count difference more than 1 should have been filtered out'


    adjustedGameLogits = gameLogits

    for sts, grp in unevenGames.groupby('smallTeamSize', sort=False):
        nameSuffix = f'(sts={sts})'

        isLastClass = sts >= len(advantageClasses)
        iAdvc = (len(advantageClasses)-1) if isLastClass else (sts-1)

        weightedSigns = pt.constant((grp.sign_largeTeamWin * len(advantageClasses) / sts) if isLastClass else grp.sign_largeTeamWin,
            name='weightedSigns'+nameSuffix)

        if isByBPST:
            highRatingness = getHighRatingnessOfBestPlayerOfSmallTeam(da, ratings, nameSuffix=nameSuffix, sts=sts, grp=grp,
                ratingClassCenter=ratingClassCenter, ratingClassSharpness=ratingClassSharpness)
            adjustment = weightedSigns * (epadc[iAdvc,0] * (1-highRatingness) + epadc[iAdvc,1] * highRatingness)
            # adjustment = pm.Deterministic('adjustment'+nameSuffix, adjustment)
        else:
            adjustment = weightedSigns * epadc[iAdvc]

        gameIxs = pt.constant(grp.gameIx, name=f'gameIxs'+nameSuffix)
        # gameIxs = pm.Deterministic('gameIxs'+nameSuffix, gameIxs)

        adjustedGameLogits = pt.inc_subtensor(adjustedGameLogits[gameIxs], adjustment, ignore_duplicates=True)

    return adjustedGameLogits

def getHighRatingnessOfBestPlayerOfSmallTeam(\
    da : PreprocessedData, 
    ratings : RatingVector,
    nameSuffix : str,
    sts : int ,
    grp : DataFrame,
    ratingClassCenter : pt.TensorVariable,
    ratingClassSharpness : pt.TensorVariable
    ) -> pt.TensorVariable:
    """
    Must be called in `with model:` block.

    Parameters: 
        sts : Small team size
        grp : The games having this small team size
    """

    gp = da.gamePlayers[['won']].join(grp[['day','sign_largeTeamWin']], on='battle_id', how='right', validate='m:1')

    gp = gp[gp.won == (gp.sign_largeTeamWin == -1) ]

    assert (gp.index.get_level_values('battle_id') == np.repeat(grp.index, sts)).all()

    smallerTeamPlayerDays = pd.MultiIndex.from_arrays([gp.index.get_level_values('player'), gp.day.array])

    smallerTeamRatingIxs = pt.constant(ratings.ixLookup.loc[smallerTeamPlayerDays], name='smallerTeamRatingIxs'+nameSuffix)

    assert len(smallerTeamRatingIxs.data) == len(grp) * sts

    smallerTeamRatings : pt.TensorVariable = ratings.vector[smallerTeamRatingIxs]

    smallerTeamBestPlayerRatings = smoothMaximum(smallerTeamRatings.reshape((len(grp),sts)), axis=1) if sts>1 else smallerTeamRatings
    # smallerTeamBestPlayerRatings.name = 'smallerTeamBestPlayerRatings'+nameSuffix

    # Optional: turn `smallerTeamBestPlayerRatings` into a model variable for debugging
    dimName = 'unevenGame'+nameSuffix
    pm.modelcontext(None).add_coord(dimName, grp.index)
    smallerTeamBestPlayerRatings = pm.Deterministic('smallerTeamBestPlayerRatings'+nameSuffix, smallerTeamBestPlayerRatings, dims=(dimName,))

    highRatingness = pt.sigmoid((smallerTeamBestPlayerRatings-ratingClassCenter)*ratingClassSharpness)
    highRatingness = pm.Deterministic('highRatingness'+nameSuffix, highRatingness, dims=(dimName,))
    return highRatingness

def makeOutcomes(da : PreprocessedData, ratings : RatingVector, gameLogits : pt.TensorVariable, firstDayRatingIndices : Optional[Series], usePotential):
    """
    Must be called in `with model:` block.

    Params:

    firstDayRatingIndices : Only if using separate virtual games
    """

    gameCountReal = len(da.games)
    separateVirtualGames = firstDayRatingIndices is not None
    gameCountEff = gameCountReal  + (0 if separateVirtualGames else 2 * len(da.players))

    def mkBernoulliObservedTrue(name, logit_p, count, dims = None):
        if usePotential:
            pm.Potential(name, -pt.sum(pt.softplus(-logit_p)))
        else:
            pm.Bernoulli(name,
                logit_p = logit_p,
                # observed = makeConst(1, count),
                observed = pt.constant(np.ones(count),'const1'),
                dims=dims
                )

    outcomes = mkBernoulliObservedTrue('Game outcomes', gameLogits, gameCountEff)

    if separateVirtualGames:
        firstDayRatings = ratings.vector[firstDayRatingIndices]
        mkBernoulliObservedTrue('virtWins', firstDayRatings, len(da.players), dims='player')
        mkBernoulliObservedTrue('virtLoss', -firstDayRatings, len(da.players), dims='player')



def createRatingsToGameLogitsMatrix(
        da : PreprocessedData, 
        ratingIxLookup : Series,
        ratingCount : int,
        dtype, 
        selfCheck : bool,
        separateVirtualGames : bool) -> Tuple[scs.spmatrix, Optional[Series]]:
    """ Creates a matrix which takes as input the vector of ratings (rating for each player-day)
        and outputs the vector of winner-loser rating differences for each game 
        (possibly including virtual wins/losses).

        If `separateVirtualGames`, a vector containing the positional index of the first day
        rating of each player is also returned.

        Params:
        -------
        ratingIxLookup :
            Index : player, day
            Values : Positional indices into the ratings vector
    """

    games = da.games.assign(gameIx = range(len(da.games)))
    gamePlayers = da.gamePlayers
    players = da.players

    gameCountEff = len(games) + (0 if separateVirtualGames else 2 * len(da.players))

    gp = gamePlayers[['won']].join(games[['gameIx','day','winnerCount','loserCount']], on='battle_id')

    # Number of nonzero entries the result matrix will have
    cooCount = len(gp) + (0 if separateVirtualGames else 2*len(players))

    ratingCoefs = np.empty((cooCount,), dtype=dtype)
    gameIxs = np.empty((cooCount,), dtype='int32')
    ratingIxs = np.empty((cooCount,), dtype='int32')

    # Number of matrix entries filled so far
    i = 0

    ratingCoefs[:len(gp)] = (1/gp.winnerCount).where(gp.won, -1/gp.loserCount)
    gameIxs[:len(gp)] = gp.gameIx
    ratingIxs[:len(gp)] = ratingIxLookup.loc[pd.MultiIndex.from_arrays(\
        [gp.index.get_level_values('player'), gp.day],
        sortorder=None)]

    firstDayIndices = ratingIxLookup.groupby(level='player').first()
    if not separateVirtualGames:
        # Add the virtual win and loss for each player
        cooSlice = slice(len(gp),None)

        ratingCoefs[cooSlice] = np.tile([1,-1], len(da.playerDays))
        gameIxs[cooSlice] = np.arange(len(games), len(games)+2*len(players))
        ratingIxs[cooSlice] = np.repeat(firstDayIndices, 2)

    mat = scs.coo_matrix((ratingCoefs,(gameIxs,ratingIxs)), shape=(gameCountEff, ratingCount)).tocsr()

    assert mat.shape[0] == gameCountEff
    assert mat.shape[1] == ratingCount

    if selfCheck:
        for x in gamePlayers.index:
            ig = games.index.get_loc(x[0])
            day = games.day.iat[ig]
            rIx = ratingIxLookup.at[(x[1],day)]
            if(gamePlayers.won[x]):
                assert abs(mat[ig, rIx] - 1/games.winnerCount.iat[ ig ]).max() < 1e-6
            else:
                assert abs(mat[ig, rIx] - -1/games.loserCount.iat[ ig ]).max() < 1e-6
        
        for j, p in enumerate(players.index):
            firstDayOfPlayer = ratingIxLookup.at[p].iat[0]
            if separateVirtualGames:
                assert firstDayIndices.iat[j] == firstDayOfPlayer
            else:
                assert mat[len(games)+2*j, firstDayOfPlayer] == 1
                assert mat[len(games)+2*j+1, firstDayOfPlayer] == -1
            

    # return pts.constant(coo.tocsc())
    return (mat, firstDayIndices if separateVirtualGames else None)

def setup_pytensor():
    pt0.config.print_global_stats = True
    pt0.config.profile = False
    pt0.config.profile_optimizer = False
    pt0.config.floatX = 'float32'
    pt0.config.openmp = False
    # pt0.config.lib__amblibm = True

    # cxxflags : str = pt0.config.gcc__cxxflags
    # if cxxflags is None or not ('AMD/aocl' in cxxflags):
    #     pt0.config.gcc__cxxflags = f"{cxxflags} -L /opt/AMD/aocl/aocl-linux-gcc-4.0/lib -I /opt/AMD/aocl/aocl-linux-gcc-4.0/include"

    # pt0.config.blas__ldflags = '-lf77blas -latlas -lgfortran'

    # pt0.config.config_print(None)

import pytensor.graph.basic as ptgb

TNode = typing.TypeVar('TNode', bound=ptgb.Node)

def named(self : TNode , name:str) -> TNode:
    self.name = name
    return self

ptgb.Node.named = named


def smoothMaximum(x : pt.TensorVariable, a=1000, axis=None) -> pt.TensorVariable:
    # return pt.logsumexp(x * a, axis=axis) / a
    m = pt.max(x, axis=axis)
    return m + pt.logsumexp((x - pt.expand_dims(m, axis=axis)) * a, axis=axis) / a

def mkPlayerDayCoord(playerDayIndex : pd.MultiIndex, playerNames : pat.Series[str]):
    assert playerDayIndex.names[0] == 'player'
    assert playerDayIndex.names[1] == 'day'
    return playerDayIndex.map(lambda t: "%s, %s" % (playerNames.at[t[0]], t[1]))

class RatingsKind(enum.Enum):
    infered = 1,
    fixed_to_mean = 2,
    fixed_to_map = 3,

RatingsKind.infered.titlePart = 'Ratings & advantage jointly fitted'
RatingsKind.fixed_to_mean.titlePart = 'Ratings fixed to mean of standard WHR'
RatingsKind.fixed_to_map.titlePart = 'Ratings fixed to MAP of standard WHR'

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class WHRModel:
    model : pm.Model
    modelDescription : dict
    ratingIxLookup : pat.Series[int]
    da : PreprocessedData
    idata : Optional[az.InferenceData] =field(init=False, default=None)
    MAP_ratings : Optional[pd.DataFrame] =field(init=False, default=None)
    extraPlayerAdjustment : ExtraPlayerAdvantageType
    ratingsKind : RatingsKind
    fileBasename : str =field(init=False)

    def __post_init__(self):
        minDate = self.da.games.day.min()
        maxDate = self.da.games.day.max()
        self.fileBasename = f'ratings_{self.ratingsKind.name}__epad_{self.extraPlayerAdjustment.name}__{minDate:%F}_to_{maxDate:%F}_{filterDescriptionFilenameFragment}.netcdf'

    def title(self) -> str:
        if self.ratingsKind == RatingsKind.infered and self.extraPlayerAdjustment in (ExtraPlayerAdvantageType.NONE, ExtraPlayerAdvantageType.DUMMY):
            return "Unadjusted WHR model"
        return f'{self.ratingsKind.titlePart}, {self.extraPlayerAdjustment.titlePart}'


    def sample(self, draws=1000, init : str = 'auto', chains = None, cores = 4, **kwargs) -> az.InferenceData:
        # nuts_sampler='blackjax'
        nuts_sampler='pymc'

        self.modelDescription['nuts_sampler'] = nuts_sampler
        self.modelDescription['init'] = init

        with self.model:
            idata = pm.sample(draws=draws, chains = chains, cores = cores, nuts_sampler=nuts_sampler, init = init, **kwargs)


        self.idata = idata

        if(self.ratingsKind == RatingsKind.infered): # Perf not an issue for the fixed-ratings models
            print('Sampling completed; extending perf stats')
            self.extendPerfStats(idata)

        return idata

    def defaultFileName(self, fileNameOverride : Optional[str] = None):
        return f'cache/{self.fileBasename}' if fileNameOverride is None else fileNameOverride

    def saveIData(self, fileNameOverride : Optional[str] = None):
        fileName = self.defaultFileName(fileNameOverride)
        self.idata.to_netcdf(fileName)

    def loadIData(self, fileNameOverride : Optional[str] = None) -> 'WHRModel':
        fileName = self.defaultFileName(fileNameOverride)

        idata = az.InferenceData.from_netcdf(fileName)
        self.idata = idata

        # assert (unadjModel.model.coords['playerDay'] == unadjModel.idata.posterior.coords['playerDay']).all()

        if self.ratingsKind == RatingsKind.infered:
            playerDaysOrderCheck = pd.DataFrame(dict(
                    model = self.model.coords['playerDay'], 
                    idata = self.idata.posterior.coords['playerDay']),
                index = self.ratingIxLookup)

            assert playerDaysOrderCheck[playerDaysOrderCheck.model != playerDaysOrderCheck.idata].idata.str.startswith('nan, ').all()

        return self

    def MAPFileName(self, fileNameOverride : Optional[str] = None):
        return f'cache/MAP_{self.fileBasename}' if fileNameOverride is None else fileNameOverride

    def calcAndSaveMAP(self):
        map1 = pm.find_MAP(model=self.model, return_raw=False)
        self.MAP_ratings = pd.DataFrame({'rating':map1['ratings']}, index=self.ratingIxLookup.index.sort_values())
        self.MAP_ratings.to_pickle(self.MAPFileName())
        return self.MAP_ratings

    def loadMAP(self) -> 'WHRModel':
        self.MAP_ratings = pd.read_pickle(self.MAPFileName())
        return self


    def extendPerfStats(
        self,
        idata : az.InferenceData, 
        filePath='perfStats.pickle'):

        perfRow : DataFrame = DataFrame(idata.sample_stats.attrs, index=[0]) #.to_frame().transpose().convert_dtypes()
        post : xa.Dataset = idata.posterior
        perfRow['chains'] = post.dims['chain']
        perfRow['draws'] = post.dims['draw']
        perfRow = pd.concat([perfRow, DataFrame(self.modelDescription, index=[0])], axis=1)
        perfRow['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS')
        # idata.sample_stats.to_dataframe()

        if 'ratings' in post.variables:
            ratings : xa.DataArray = post.ratings

            examplePlayerIndices = self.ratingIxLookup[505371]
            exampleRatings = naturalRatingToElo(ratings[:,:,[examplePlayerIndices.iat[0], examplePlayerIndices.iat[-1]]])
            exampleMeans = exampleRatings.mean(['chain','draw'])
            exampleSDevs = exampleRatings.std(['chain','draw'])
            perfRow['p505371_start_mean'] = exampleMeans.values[0]
            perfRow['p505371_start_sdev'] = exampleSDevs.values[0]
            perfRow['p505371_end_mean'] = exampleMeans.values[1]
            perfRow['p505371_end_sdev'] = exampleSDevs.values[1]
            perfRow['first_date'] = self.da.games.started.min()

        perfRow['installed_libs'] = 'libatlas-base-dev'
        perfRow['ptconfig.blas__ldflags'] = pt0.config.blas__ldflags
        perfRow['ptconfig.lib__amblibm'] = pt0.config.lib__amblibm

        # Move columns to end
        for col in ['arviz_version','inference_library','inference_library_version','tuning_steps']:
            try:
                x = perfRow.pop(col)
                perfRow[col] = x
            except KeyError:
                pass

        perfRow = perfRow.convert_dtypes()

        with open(filePath,'ab') as f:
            perfRow.to_pickle(f)

    def compute_log_likelihood(self):
        pm.compute_log_likelihood(self.idata, model=self.model, extend_inferencedata=True)

    def ratingsByPlayerDay(self) -> xa.DataArray:
        ratingIxLookup = self.ratingIxLookup
        postRatings : xa.DataArray = self.idata.posterior['ratings']
        postRatings = \
            postRatings \
                .assign_coords(player = ('playerDay', ratingIxLookup.index.get_level_values('player'))) \
                .assign_coords(day = ('playerDay', ratingIxLookup.index.get_level_values('day'))) \
                .set_xindex(['player', 'day'])

        return postRatings.assign_coords(name = ('playerDay', self.da.players.name[postRatings.coords['player']] ))

    def getMeanRatingsForPlayerDays(self, playerDays : pd.MultiIndex) -> pat.Series[float]:
        assert playerDays.names == ('player', 'day'), playerDays.names

        return pat.Series[float](
            self.idata.posterior['ratings'].mean(['chain','draw']).isel(playerDay = self.ratingIxLookup[playerDays]),
            index=playerDays)

    def epad_in_Elo(self, bpstElo = None):
        assert self.extraPlayerAdjustment != ExtraPlayerAdvantageType.NONE, self.extraPlayerAdjustment
        epad = eloPerNaturalRating * self.idata.posterior['epadc']
        if(bpstElo is not None):
            t = eloHighness(self.idata, bpstElo)
            epad = (1-t) * epad.sel(ratingclass='low rating') + t * epad.sel(ratingclass='high rating')
        epad.name = 'Extra player advantage in Elo'
        return epad

    def ensureHaveLogLikelihood(self):
        if not 'log_likelihood' in self.idata:
            with self.model:
                pm.compute_log_likelihood(self.idata)

def titles(models : typing.Sequence[WHRModel]) -> list[str]:
    return [m.title() for m in models]