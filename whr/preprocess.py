from whr.basic import *
from whr.read_data import *

@version(3)
def gamesToPlayerDays(games : DataFrame, gamePlayers : DataFrame) -> DataFrame:
    """
    Parameters
    ----------
    games : DataFrame containing columns 'day', 'winnerCount', 'loserCount'

    """

    gp = gamePlayers.join(
            games[['day']].assign(winnerWeight = 1/games.winnerCount, loserWeight = 1/games.loserCount), 
            on='battle_id', validate='m:1')

    gp['weight'] = gp.winnerWeight.where(gp.won, gp.loserWeight)

    playerDays = gp[['day','weight']].groupby(['player','day'], sort=True).agg({'weight':'sum'})

    firstDaySDev = 2000 * naturalRatingPerElo

    def processPlayerDays(grp : DataFrame):
        varFromGames = np.empty((len(grp),), dtype=float)
        varFromDays = np.empty((len(grp),), dtype=float)
        sdev = np.empty((len(grp),), dtype=float)

        varFromGames[0] = -1
        varFromDays[0] = -1
        sdev[0] = firstDaySDev

        if len(grp) > 1:
            weights = grp.weight.to_numpy()
            varFromGames [1:] = weights[1:] * zkNaturalRatingVariancePerGame

            dayDiffs = np.diff(grp.index.get_level_values('day').to_numpy('datetime64[D]')).astype(np.int32, copy=False)
            
            assert len(dayDiffs) == len(grp) - 1
            assert dayDiffs[0] == (grp.index[1][1] - grp.index[0][1]).days

            totalWeights = np.cumsum(weights)

            varFromDays[1:] = dayDiffs * zkNaturalRatingVariancePerDay(totalWeights[:-1])

            sdev[1:] = np.sqrt(varFromGames[1:] + varFromDays[1:])
                                

        return grp.copy().assign(
                varFromGames = varFromGames,
                varFromDays = varFromDays,
                sdev = sdev,
                playerDayCount=len(grp),
                )

    playerDays = playerDays.groupby(level="player", group_keys=False).apply(processPlayerDays)
    # playerDays.set_index(['playerDayCount',playerDays.index], inplace=True)
    # playerDays.sort_index(inplace=True)

    # Add human-readable scale for display
    playerDays['elo_sdev'] = playerDays.sdev * eloPerNaturalRating

    return playerDays

        # reordIx: int, Positional index into `reordPlayerDays` 
        # extIx: int, Positional index into `extPlayerDays` 

class GamesSchema(DataFrameModel):
    battle_id : pat.Index[int] = pa.Field()

    day : pat.Series[np.datetime64] = pa.Field()
    winnerCount : pat.Series[np.int8] = pa.Field()
    loserCount : pat.Series[np.int8] = pa.Field()

class GamePlayersSchema(DataFrameModel):
    battle_id : pat.Index[int] = pa.Field()
    player : pat.Index[int] = pa.Field()

    won : pat.Series[bool] = pa.Field()

class PlayerDaysSchema(DataFrameModel):
    player : pat.Index[int] = pa.Field()
    day : pat.Index[np.datetime64] = pa.Field()

    sdev : pat.Series[float] = pa.Field()
    """Standard deviation of natural rating change compared to previous day (prior)"""

class PlayersSchema(DataFrameModel):
    player : pat.Index[int] = pa.Field()

    name : pat.Series[str] = pa.Field()

    

@dataclass()
class PreprocessedData:
    games : pat.DataFrame[GamesSchema]

    gamePlayers : pat.DataFrame[GamePlayersSchema]

    playerDays : pat.DataFrame[PlayerDaysSchema]

    players : pat.DataFrame[PlayersSchema]

    @staticmethod
    def load() -> "PreprocessedData":
        games, gamePlayers = cached('games', readAndFilterZklaBattles, DEFAULT_MONTHS)
        # games = pat.DataFrame[GamesSchema](games)
        # gamePlayers = pat.DataFrame[GamePlayersSchema](gamePlayers)
        playerNames = cached('playerNames', readZklaPlayerNames)
        # playerNames = pat.DataFrame[PlayerDaysSchema](playerNames)
        playerDays = cached('playerDays', gamesToPlayerDays, games, gamePlayers)
        assert playerDays.index.is_monotonic_increasing

        players = playerDays.value_counts('player', sort=False).to_frame('playerDayCount').join(playerNames, how='left')

        namelessPlayers = players.index[players.name.isna()]
        players.loc[namelessPlayers, 'name'] = [f'[Unnamed player with ID {p}]' for p in namelessPlayers]

        return PreprocessedData(games=games, gamePlayers=gamePlayers, playerDays=playerDays, players=players)

    def restrictGames(self, gameBools : pat.Series[bool], restrictPlayerDays : bool) -> "PreprocessedData":
        games = self.games[gameBools]
        gamePlayers =self.gamePlayers.loc[games.index]

        if restrictPlayerDays:
            playerDays = self.playerDays.loc[\
                gamePlayers.join(games[['day']], on='battle_id').set_index('day',append=True).droplevel('battle_id').index.drop_duplicates()
            ].sort_index()
        else:
            playerDays = self.playerDays

        return PreprocessedData(games=games, 
            gamePlayers=gamePlayers,
            playerDays=playerDays,
            players=self.players)

    def getPlayersWhoOftenPlayInUnevenGames(self, maxTeamSize=6, minTotal=50):
        return self.gamePlayers.join(
            self.games.loc[lambda d: (d.winnerCount != d.loserCount) & (d.winnerCount <= maxTeamSize) & (d.loserCount <= maxTeamSize)],
            how='right', on='battle_id', rsuffix='game_', validate='m:1') \
            .assign(nSmall = lambda d: d.won == (d.winnerCount < d.loserCount))[['nSmall']] \
            .assign(nLarge = lambda d: ~d.nSmall) \
            .groupby('player').sum() \
            .join(self.players, on='player', validate='m:1')[['name','nSmall','nLarge']]  \
            .join(self.gamePlayers.groupby('player').size().to_frame('nTotal'), on='player') \
            [lambda d: d.nTotal >= minTotal] \
            .assign(**{'(nSmall-nLarge)/nTotal' : lambda d: (d.nSmall - d.nLarge)/d.nTotal}) \
            .sort_values('(nSmall-nLarge)/nTotal')

    def getGamesWithPlayer(self, player) -> pat.DataFrame[GamesSchema]:
        return self.games.join(self.gamePlayers.swaplevel().loc[player], how='right', rsuffix='gamePlayer_')

    def getUnevenGamesWithSmallTeamSize(self, sts) -> pat.DataFrame[GamesSchema]:
        return self.games[lambda d: \
                (d.loserCount == sts) & (d.winnerCount > sts) | \
                (d.loserCount > sts) & (d.winnerCount == sts)]

    def getUnevenGames(self) -> pat.DataFrame[GamesSchema]:
        return self.games[self.games.winnerCount != self.games.loserCount].assign(smallWon = lambda d: d.winnerCount < d.loserCount)

    def getUnevenGamePlayers(self) -> pat.DataFrame:
        return self.gamePlayers.drop(columns='team_won', errors='ignore').join(self.getUnevenGames(), how='right', validate='m:1', on='battle_id')

    def getSmallerTeamPlayers(self, maxSmallTeamSize=None) -> pat.DataFrame:
        d = self.getUnevenGamePlayers()[lambda d: d.won == d.smallWon]
        if maxSmallTeamSize is not None: d = d[(d.winnerCount <= maxSmallTeamSize) | (d.loserCount <= maxSmallTeamSize)]
        return d


