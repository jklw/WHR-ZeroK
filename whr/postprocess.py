from whr.models_shared import *
from whr.imports import *

def getElo(
        self: WHRModel,
        player: int,
        day: Union[np.datetime64, int],
        ):
    idata = self.idata

    if day is np.datetime64:
        try:
            ipday = self.ratingIxLookup.loc[(player,day)]
        except KeyError:
            logging.error(f'Player only has these days: {self.ratingIxLookup.loc[player].index}')
            raise
    else:
        ipday = self.ratingIxLookup.loc[player][day]

    coord = idata.posterior.coords["playerDay"][ipday].values

    rat = idata.posterior.ratings.isel(playerDay = ipday)
    elo = naturalRatingToElo(rat)

    # elo.name = f'Posterior Elo of {coord}'
    elo.name = f'Elo {str(coord).replace(" 00:00:00", "")}'
    return (elo, coord)

WHRModel.getElo = getElo

def comparePlayerDay(
        models: list[WHRModel],
        player: int,
        day: Union[np.datetime64, int],
        data_labels=['adjusted model', 'unadjusted model'], 
        ax=None):

    ys = [m.getElo(player,day) for m in models]

    assert [y == ys[0][1] for y in ys[1:]]

    ax = az.plot_density([y[0] for y in ys], hdi_prob=0.999, data_labels=data_labels, ax=ax)
    return ax

def getBPSTElo(da : PreprocessedData, ratings : pat.Series[float]) -> pat.Series[float]:

    smallTeamRatings = da.getSmallerTeamPlayers()[['day']].join(ratings.to_frame('rating'), on=('player','day'), validate='m:1')

    return smallTeamRatings.groupby('battle_id').rating.max().pipe(naturalRatingToElo)

def createGameCountByBPSTSummaryTable(da : PreprocessedData, ratings : pat.Series[float]):
        # .assign(highness = lambda d: models_shared.eloHighness(models['meanBPST'].idata, d.BPST_Elo)) \
    gameCounts = getBPSTElo(da, ratings).to_frame("BPST_Elo") \
        .join(da.games[['winnerCount','loserCount']]) \
        .assign(
            advc = lambda d: advantageClasses[d.winnerCount.clip(upper=d.loserCount).clip(upper=len(advantageClasses))-1],
            smallWon = lambda d: d.winnerCount < d.loserCount,
            # cat = lambda d: pd.cut(d.highness, bins=3, labels=['low','mid','high'])
            cat = lambda d: pd.cut(d.BPST_Elo, bins=[-math.inf,1800,2200,math.inf],include_lowest=True, 
                labels=['< 1800 Elo','[1800; 2200]','> 2200 Elo']
            )
        )

    gc2 : pd.DataFrame    
    gc2 = gameCounts.groupby(['advc','cat', 'smallWon'])\
        .pipe(lambda d: typing.cast(pd.DataFrame, typing.cast(pd.Series, d.size()).to_frame('count'))
            # .assign(highness = d.highness.sum()/d.size())
            ) \
        .unstack('smallWon') \
        .rename(columns = {True:'Small', False : 'Large', 'count':'↓ winner ↓'})

    gc2.index.set_names(['Size', 'Rating of best player of small team →'], inplace=True)
    gc2.columns.set_names(None, level=1, inplace=True)
    return gc2.unstack( 'Rating of best player of small team →').reorder_levels([2,0,1], axis=1).sort_index(axis=1)


def putLegendOnTopOfFigure(ax : plt.Axes, bbox: Tuple[float,float]):
    l : plt.Legend = ax.get_legend()
    texts = l.texts
    legendHandles = l.legendHandles
    ax.legend([])
    ax.figure.legend(legendHandles[::-1], [t.get_text() for t in texts][::-1], loc='upper center', bbox_to_anchor=bbox)