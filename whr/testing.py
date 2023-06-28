from whr.basic import *
from whr.models_shared import *

def validateFixedRatingsBPST(idata : az.InferenceData, unevenGames : PreprocessedData, unevenGamesFixedRatings : pat.Series[float]):
    seenGameCount=0

    for varName in idata.posterior:
        m = re.match('^smallerTeamBestPlayerRatings\(sts=(\d+)\)', varName)
        if m is None: continue
        sts = int(m[1])
        bpstRatings : xa.DataArray = idata.posterior[varName]
        actualRatingHighnesses = idata.posterior[f'highRatingness(sts={sts})']
        battleIds : xa.DataArray = idata.posterior[f'unevenGame(sts={sts})']

        seenGameCount += len(battleIds)
        assert (battleIds == unevenGames.getUnevenGamesWithSmallTeamSize(sts).index).all()

        games = unevenGames.games.loc[battleIds].assign(smallWon = lambda d: d.winnerCount == sts)[['smallWon', 'day']]
        smallTeamPlayers = unevenGames.gamePlayers.join(games, how='right')[lambda d: d.won == d.smallWon]
        assert len(smallTeamPlayers) == len(battleIds) * sts

        smallTeamPlayerRatings = smallTeamPlayers.join(unevenGamesFixedRatings.to_frame('rating'), on = ('player', 'day'))

        expectedBPSTRatings = smallTeamPlayerRatings.groupby('battle_id').rating.max()
        assert (expectedBPSTRatings.index == battleIds).all()

        # Ratings are fixed in this model
        assert (bpstRatings.min(dim=['chain','draw']) == bpstRatings.max(dim=['chain','draw'])).all()
        assert (actualRatingHighnesses.min(dim=['chain','draw']) == actualRatingHighnesses.max(dim=['chain','draw'])).all()

        bpstCheck = expectedBPSTRatings.to_frame('expected').assign(actual = bpstRatings[0,0], err = lambda d: d.actual - d.expected)

        # Some positive difference is expected here due to the smooth maximum
        print(f'BPST min/max difference: {bpstCheck.err.min():.5f}, {bpstCheck.err.max():.5f}')
        assert np.allclose(bpstCheck.expected, bpstCheck.actual, atol=1e-2, rtol=0)

        highnessCheck = ratingHighness(idata, expectedBPSTRatings).to_frame('expected').assign(actual = actualRatingHighnesses[0,0], err = lambda d: d.actual - d.expected)

        assert np.allclose(highnessCheck.expected, highnessCheck.actual, atol=1e-2, rtol=0)

        
    assert seenGameCount == len(unevenGames.games)

def generateTestData(gameCount : int, rng : Generator) -> Tuple[ DataFrame, DataFrame ]:
    # Ground truth for testing, and a sample of games derived from it
    trueRatingsNested = pd.DataFrame([
        ('consty0',  {  0:0,  1:0,    2:0,    3:0}), 
        ('consty-1', {  0:-1, 1:-1,   2:-1,   3:-1}), 
        ('consty1',  {  0:1,  1:1,    2:1,    3:1}), 
        ('improver', {  0:-1, 1:0,    2:1,    3:2}), 
        ('fast_impr',{  0:-2, 1:0,    2:2,    3:4}), 
        ('0:5 3:4',  {  0:5,                  3:4}),
        ('1:4 3:5',  {        1:4,            3:5}),
        ('lob0', {0:-1}),
        ('lob1', {1:-1}),
        ('lob2', {2:-1}),
        ('lob3', {3:-1}),
    ],
        columns=["player","ratings"]
    )

    trueRatingsNested.set_index('player', inplace=True)
    trueRatingsNested : pd.Series = trueRatingsNested['ratings']

    trueRatingsLong = trueRatingsNested.map(lambda days: days.items()).explode().apply(lambda t: pd.Series(t))
    trueRatingsLong.columns = ['day','rating']
    trueRatingsLong.set_index('day', inplace=True, append=True)

    trueRatingsLong.assign(elo = naturalRatingToElo(trueRatingsLong.rating).round().convert_dtypes())

    return trueRatingsLong, generateGames(gameCount, trueRatingsLong, rng)


def generateGames(gameCount : int, trueRatingsLong : DataFrame, rng : Generator):
    playersByDay : pd.Series = trueRatingsLong.rating.groupby(level="day").apply(lambda df: df.index.get_level_values('player')).loc[lambda s: s.map(lambda ps: len(ps)>1)]

    games = pd.DataFrame({"day":rng.integers(len(playersByDay), size=gameCount)})
    # gameAvailablePlayers = playersByDay.iloc[games.day].reset_index(drop=True)
    gamePCounts = rng.integers(
        low = 2,
        high = np.fromiter((len(playersByDay[day]) for day in games.day), dtype=int, count=gameCount),
        endpoint=True
    )

    team1s = np.empty((gameCount,), dtype = object)
    team2s = np.empty((gameCount,), dtype = object)
    team1rs = np.empty((gameCount,), dtype = 'float64')
    team2rs = np.empty((gameCount,), dtype = 'float64')

    # Generate teams
    for g in range(gameCount):
        day = games.day.iloc[g]
        availablePlayers = playersByDay[day]
        n = gamePCounts[g]
        players = rng.choice(len(availablePlayers), size=n, replace=False)
        team1 = availablePlayers[players[:n//2]].array
        team2 = availablePlayers[players[n//2:]].array
        team1s[g] = team1
        team2s[g] = team2
        team1rs[g] = np.mean(trueRatingsLong.rating[((p, day) for p in team1)])
        team2rs[g] = np.mean(trueRatingsLong.rating[((p, day) for p in team2)])

    probTeam1Win = 1 / (1 + np.exp(team2rs - team1rs))
    isTeam1Win = rng.binomial(1, probTeam1Win)

    games["winner"] = np.choose(isTeam1Win, [team2s, team1s])
    games["loser"]  = np.choose(isTeam1Win, [team1s, team2s])
    return games
