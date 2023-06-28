from whr.model_cumsum_matrix import *

def makeFixedRatingsModel(da : PreprocessedData, 
    ratings : pat.Series[float],
    ratingsKind : RatingsKind,
    epadType : ExtraPlayerAdvantageType) -> WHRModel:

    assert ratingsKind == RatingsKind.fixed_to_map or ratingsKind == ratingsKind.fixed_to_mean, (ratingsKind, type(ratingsKind))
    assert isinstance(epadType, ExtraPlayerAdvantageType), (epadType, type(epadType))

    ratingIxLookup = pat.Series[int](range(len(ratings)), index=ratings.index)

    frModel = pm.Model(coords=mkCoords(da), check_bounds=False)


    with frModel:
        ratings = pm.ConstantData('ratings',ratings, dims="playerDay")

        ratingVector = RatingVector(ratings, ratingIxLookup)

        ratingsToGameLogitsMatrix, firstDayIndices = createRatingsToGameLogitsMatrix(
            da,
            ratingIxLookup = ratingIxLookup,
            ratingCount = len(da.playerDays),
            dtype = pt0.config.floatX,
            selfCheck = False, separateVirtualGames=True)

        ratingsToGameLogits = pts.constant(ratingsToGameLogitsMatrix, name='ratingsToGameLogits')

        ratingsColVector = ratings.dimshuffle(0,'x')
        gameLogits = pts.structured_dot(ratingsToGameLogits, ratingsColVector).dimshuffle(0)

        gameLogits.name = 'gameLogits (before adjustment)'

        # WHR_mult = pm.Normal('WHR_mult', mu=1, sigma=1)
        WHR_mult = pm.ConstantData('WHR_mult', 1)

        adjustedGameLogits = makeExtraPlayerAdvantageVariables(da, ratings=ratingVector, gameLogits=gameLogits*WHR_mult, typ=epadType)
        adjustedGameLogits.name = 'adjustedGameLogits'

        pm.Bernoulli('Game outcomes',
            logit_p = adjustedGameLogits,
            # observed = makeConst(1, count),
            observed = pt.constant(np.ones(len(da.games)),'const1'),
            )

        return WHRModel(frModel, {}, ratingIxLookup=ratingIxLookup, da=da, extraPlayerAdjustment=epadType, ratingsKind=ratingsKind)

def makeAllFixedRatingsModels(unevenGamesData : PreprocessedData,  unevenGamesMeanRatings : pat.Series[float],  unevenGamesMAPRatings : pat.Series[float] ) -> dict[str,WHRModel]:
    return dict(
    meanNull = makeFixedRatingsModel(unevenGamesData, unevenGamesMeanRatings, RatingsKind.fixed_to_mean, ExtraPlayerAdvantageType.DUMMY),
    meanSimple = makeFixedRatingsModel(unevenGamesData, unevenGamesMeanRatings, RatingsKind.fixed_to_mean, ExtraPlayerAdvantageType.BY_TEAM_SIZE),
    meanBPST = makeFixedRatingsModel(unevenGamesData, unevenGamesMeanRatings, RatingsKind.fixed_to_mean, ExtraPlayerAdvantageType.BY_TEAM_SIZE_AND_BPST),
    mapNull = makeFixedRatingsModel(unevenGamesData, unevenGamesMAPRatings, RatingsKind.fixed_to_map, ExtraPlayerAdvantageType.DUMMY),
    mapSimple = makeFixedRatingsModel(unevenGamesData, unevenGamesMAPRatings, RatingsKind.fixed_to_map, ExtraPlayerAdvantageType.BY_TEAM_SIZE),
    mapBPST = makeFixedRatingsModel(unevenGamesData, unevenGamesMAPRatings, RatingsKind.fixed_to_map, ExtraPlayerAdvantageType.BY_TEAM_SIZE_AND_BPST),
    )