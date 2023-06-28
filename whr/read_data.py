from whr.basic import *

DEFAULT_MONTHS = np.array([str(y) for y in range(2017,2022)] + [f'2022-{i:02d}' for i in range(5,13)] + [f'2023-{i:02d}' for i in range(1,6)])
# DEFAULT_MONTHS =  np.array([f'2023-{i:02d}' for i in range(1,5)])

STATIC_DATA_DATE='2023-06-10'

def readZklaBattles(months) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Returns: games, gamePlayers
    """
    with gzip.open('data/%s-maps.json' % STATIC_DATA_DATE) as file:
        maps = pd.DataFrame(json.load(file))
    maps.set_index('map_id', inplace=True)

    monthResults = [readZklaBattlesFile(month, maps) for month in months]
    games = pd.concat([t[0] for t in monthResults])
    gamePlayers = pd.concat([t[1] for t in monthResults])

    games.sort_index(inplace=True)
    gamePlayers.sort_index(inplace=True)

    return games, gamePlayers

filterDescriptionFilenameFragment = '_autohost'

@version(11)
def readAndFilterZklaBattles(months) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Returns: games, gamePlayers
    """
    games, gamePlayers = readZklaBattles(months)

    def dropGames(dropees : pd.Index, because : str, logLevel = logging.WARNING, playersToo=True):
        if(len(dropees) > 0):
            logging.log(logLevel, f'Dropping these {len(dropees)} games because {because}: {dropees}')

        games.drop(index = dropees, inplace=True)
        if playersToo: gamePlayers.drop(index = dropees, level=0, inplace=True)

    dropGames(games.index[games.winnerCount.isna()], 'they have no players', playersToo=False)

    assert games.loserCount.notna().all()

    dropGames(games.index[games.winnerCount == 0], 'they have no winners', logging.INFO)
    dropGames(games.index[games.loserCount == 0], 'they have no losers')

    dropGames(gamePlayers.index[(gamePlayers.team_id != 1) & (gamePlayers.team_id != 2)].unique('battle_id'),
        'not all teamIds are 1 or 2')

    dropGames(games.index[abs(games.winnerCount - games.loserCount) > 1], 'they have team player count difference of more than 1 (should not happen for rated games)')

    dropGames(games.index[(games.map_is_special == 1) & (games.map_name != 'Dockside v2')], 'they are on a "special" map')

    games = games.assign(
        winnerCount = games.winnerCount.astype(np.int8),
        loserCount = games.loserCount.astype(np.int8),
        )

    return games, gamePlayers

def readZklaBattlesFile(month : str, maps : DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Returns: games, gamePlayers
    """
    with gzip.open('data/%s-battles.json' % month) as file:
        games = pd.DataFrame(json.load(file))
    games.set_index('battle_id', verify_integrity=True, inplace=True)
    # games = games[(games.is_elo != 0) & (games.is_ffa != 1) & (games.has_bots != 1) & (games.is_chicken != 1) & (games.game_id == 1)]
    preFilter = (games.is_matchmaker != 1) & (games.is_elo != 0) & (games.has_bots != 1) & (games.is_chicken != 1) & (games.game_id == 1) \
                    & (games.duration_sec >= 20) \
                    & (games.is_autohost_teams != 0)
    # games['preFilter'] = preFilter
    games = games[preFilter]

    games['started'] = pd.to_datetime(games.started_sec, unit='s')
    del games['started_sec']

    with gzip.open('data/%s-battle_player.json' % month) as file:
        gamePlayers = pd.DataFrame(json.load(file))

    gamePlayers.rename({'player_id':'player'},axis=1, inplace=True)
    gamePlayers.set_index(['battle_id','player'], verify_integrity=True, inplace=True)
    gamePlayers.sort_index(inplace=True)

    gamePlayers = gamePlayers.join(games[['team_won']], on='battle_id', how='inner', validate='m:1')
    gamePlayers['won'] = gamePlayers.team_id == gamePlayers.team_won

    games['winnerCount'] = gamePlayers.won.groupby('battle_id').sum().astype(np.int8)
    games['loserCount'] = (1-gamePlayers.won).groupby('battle_id').sum().astype(np.int8)
    # games['winnerWeight'] = 1/games.winnerCount
    # games['loserWeight'] = 1/games.loserCount
    games['day'] = games.started.to_numpy('datetime64[D]')

    games = games.join(maps.rename(lambda c: 'map_'+c, axis=1) , on='map_id', how='left', validate='m:1')


    return games, gamePlayers


@version(2)
def readZklaPlayerNames():
    with gzip.open(f'data/{STATIC_DATA_DATE}-players.json') as file:
        j = json.load(file)
    pns = pd.DataFrame({'name':j['nickname']}, index = j['player_id'])
    return pns