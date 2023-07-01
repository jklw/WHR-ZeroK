from whr.models_shared import *

def createLowerCumsumMatrix(
        da : PreprocessedData, 
        dtype) -> scs.spmatrix:
    """ Creates a matrix M such that
        `ratings = M * ratingIncrements + ratingIncrements` 
        (both input and output vector ordered according to `da.playerDays`)

        This is, for each player, a strictly lower triangular matrix filled only with 1s 
        (the main diagonal is omitted, thus the `+` above).
    """
    # Build cumsum matrix, excluding the diagonal.
    pdcs = da.players.playerDayCount

    cooCount = ((pdcs * (pdcs - 1))//2).sum()

    pd = len(da.playerDays)
    assert pd == sum(pdcs)

    ratingIxs = np.empty((cooCount,), dtype='int32')
    innovationIxs = np.empty((cooCount,), dtype='int32')

    i = 0 # filling position for above arrays
    for (p, c) in pdcs.items():
        # p : player, c : dayCount of player
        start = da.playerDays.index.get_loc(p).start
        for j in range(1, c):
            ratingIxs[i:(i+j)] = start + j
            innovationIxs[i:(i+j)] = np.arange(start=start, stop=start+j)
            i += j

    assert i == cooCount

    incrementCoeffs = np.ones((cooCount,), dtype='int8')

    return scs.coo_matrix((incrementCoeffs,(ratingIxs,innovationIxs)), shape=(pd, pd)).tocsr()

def mkCoords(da : PreprocessedData):
    return {
            "player": da.players.name, 
            # "playerDay": playerDays.index.map(lambda t: "%s, %s" % (players.name.at[t[1]], t[2])),
            "playerDay": mkPlayerDayCoord(da.playerDays.index, da.players.name),
            # "game": da.games.index
    }

def mkIncrements(da : PreprocessedData):
    playerDayRatingSDevs = pm.ConstantData('playerDayRatingSDevs ', da.playerDays.sdev , dims=("playerDay",))

    return pm.Normal('increments', mu=0, sigma = playerDayRatingSDevs, dims="playerDay")

def createCumsumMatrixModel(da : PreprocessedData, 
        separateVirtualGames : bool, 
        extraPlayerAdjustment : ExtraPlayerAdvantageType,
        preMultiplyCumsumMatrix = False,
        useOutcomesPotential = True
        ) \
    -> WHRModel :

    pdc = len(da.playerDays)

    # Trivial lookup in this model since we don't reorder playerDays
    ratingIxLookup = pat.Series[int](range(pdc), index=da.playerDays.index)

    model = pm.Model(coords=mkCoords(da), check_bounds=False)

    with model:
        increments = mkIncrements(da)

        lowerCumsumMatrix = createLowerCumsumMatrix(da, dtype = pt0.config.floatX)

        lowerCumsum = pts.constant(lowerCumsumMatrix, 'lowerCumsum')

        incrementsColVector = increments.dimshuffle(0,'x')

        ratings = pm.Deterministic('ratings', 
            increments + pts.structured_dot(lowerCumsum, incrementsColVector).dimshuffle(0),
            dims="playerDay")

        ratingVector = RatingVector(ratings, ratingIxLookup)

        ratingsToGameLogitsMatrix, firstDayIndices = createRatingsToGameLogitsMatrix(
            da,
            ratingIxLookup = ratingIxLookup,
            ratingCount = pdc,
            dtype = pt0.config.floatX,
            selfCheck = False, separateVirtualGames=separateVirtualGames)

        ratingsToGameLogits = pts.constant(ratingsToGameLogitsMatrix, name='ratingsToGameLogits')

        if preMultiplyCumsumMatrix:
            incrementsToGameLogitsMatrix = ratingsToGameLogitsMatrix @ (lowerCumsumMatrix + scs.identity(pdc, dtype='int8'))
            incrementsToGameLogits = pts.constant(incrementsToGameLogitsMatrix, name='incrementsToGameLogits ')

            gameLogits = pts.structured_dot(incrementsToGameLogits, incrementsColVector).dimshuffle(0)
        else:
            ratingsColVector = ratings.dimshuffle(0,'x')
            gameLogits = pts.structured_dot(ratingsToGameLogits, ratingsColVector).dimshuffle(0)

        gameLogits.name = 'gameLogits (before adjustment)'

        if extraPlayerAdjustment != ExtraPlayerAdvantageType.NONE:
            adjustedGameLogits = makeExtraPlayerAdvantageVariables(da, ratings=ratingVector, gameLogits=gameLogits, typ=extraPlayerAdjustment)
            adjustedGameLogits.name = 'adjustedGameLogits'
        else:
            adjustedGameLogits = gameLogits

        makeOutcomes(da, ratingVector, adjustedGameLogits, firstDayIndices, usePotential=useOutcomesPotential)


        # For perf stat logging
        modelDescription = {
            'games' : len(da.games),
            'realDays' : pdc,
            'paddingDays' : 0,
            'players' : len(da.players),
            'cumsumBlocks' : None,
            'virtualGameOutcomeCount' : 2 * len(da.players),
            'ratingsToGameLogitsMatrix.count_nonzero' : ratingsToGameLogitsMatrix.count_nonzero(),
            'ratingsToGameLogitsMatrix.type' : repr(ratingsToGameLogits),
            'floatX' : pt0.config.floatX,
            'innovsToGameLogitsMatrix.count_nonzero' : incrementsToGameLogitsMatrix.count_nonzero() if preMultiplyCumsumMatrix else None,
            'innovsToGameLogitsMatrix.type' : repr(incrementsToGameLogits) if preMultiplyCumsumMatrix else None,
            'lowerCumsumMatrix.count_nonzero' : lowerCumsumMatrix.count_nonzero(),
            'lowerCumsumMatrix.type' : repr(lowerCumsum),
            'preMultiplyCumsumMatrix' : preMultiplyCumsumMatrix,
            'separateVirtualGames ' : separateVirtualGames,
            'ptconfig.openmp' : pt0.config.openmp,
            'useOutcomesPotential' : useOutcomesPotential,
            'has_epadc' : extraPlayerAdjustment,
            'comment' : ''
        }

        return WHRModel(model, modelDescription=modelDescription, ratingIxLookup=ratingIxLookup, da=da, extraPlayerAdjustment=extraPlayerAdjustment,
                ratingsKind=RatingsKind.infered)


# Specific models with the same settings for comparison
def createUnadjustedCumsumMatrixModel(da: PreprocessedData) -> WHRModel:
    return createCumsumMatrixModel(da, separateVirtualGames=True, extraPlayerAdjustment=ExtraPlayerAdvantageType.NONE, useOutcomesPotential=False)

def createSimpleAdjustedModel(da: PreprocessedData) -> WHRModel:
    return createCumsumMatrixModel(da, separateVirtualGames=True,
                                   extraPlayerAdjustment=ExtraPlayerAdvantageType.BY_TEAM_SIZE, useOutcomesPotential=False)

def createBPSTAdjustedModel(da: PreprocessedData) -> WHRModel:
    return createCumsumMatrixModel(da, separateVirtualGames=True,
                                   extraPlayerAdjustment=ExtraPlayerAdvantageType.BY_TEAM_SIZE_AND_BPST, useOutcomesPotential=False)
