from whr.models_shared import *

@dataclass(order=True, config=ConfigDict(arbitrary_types_allowed=True))
class ProposedBlockSplit:
    priority: int
    splitIndex: int=field(compare=False) # int index of the last element going into the first fragment
    block: DataFrame=field(compare=False)

def makePlayerDayCountBlocks(playerDayCounts : Series, maxBlocks : int) -> Tuple[ DataFrame, DataFrame ]:
    assert maxBlocks > 1

    block0 = playerDayCounts.value_counts(sort=False).to_frame('playerCount')
    block0.index.name = 'dayCount'
    block0.sort_index(inplace=True)
    block0['cumCount'] = block0.playerCount.cumsum()

    totalWaste = (block0.playerCount[:-1] * (block0.index[-1] - block0.index[:-1])).sum()
    totalPlayerDays = (block0.playerCount * block0.index).sum()

    wasteLog = [totalWaste] 

    def proposeSplit(block : DataFrame) -> Optional[ProposedBlockSplit]:
        if len(block) < 2: return None

        # Current waste: sum_i playerCount[i] * (maxDayCount - dayCount[i])

        # Waste after split at k:
        # sum_{i<=k} playerCount[i] * (dayCount[k] - dayCount[i]) +
        # sum_{i>k}  playerCount[i] * (maxDayCount - dayCount[i])

        # Waste reduction:
        # sum_{i<=k} playerCount[i] * (maxDayCount - dayCount[k])
        # = (maxDayCount - dayCount[k]) * cumCount[k]

        wasteReductions = (block.index[-1] - block.index[:-1]) * block.cumCount[:-1]
        splitIndex = wasteReductions.argmax() if block.index[0] != 1 else 0 # Always split off the 1-day players first
        return ProposedBlockSplit(-wasteReductions.iat[splitIndex], splitIndex, block)


    heap : list[ProposedBlockSplit] = [proposeSplit(block0)]
    finishedBlocks : list[DataFrame] = []

    while(len(heap)>0):
        prop = heapq.heappop(heap)

        s = prop.splitIndex + 1

        block1 = prop.block.iloc[:s]
        block2 = prop.block.iloc[s:]

        block2 = block2.assign(cumCount = block2.cumCount - block1.cumCount.iat[-1])

        totalWaste += prop.priority # priority = -wasteReduction
        wasteLog.append(totalWaste)

        if len(heap) + len(finishedBlocks) + 2 >= maxBlocks:
            finishedBlocks += [block1, block2]
            finishedBlocks.extend((p.block for p in heap))
            break

        for block12 in [block1, block2]:
            prop12 = proposeSplit(block12)
            if prop12 is None: finishedBlocks.append(block12)
            else: heapq.heappush(heap, prop12)

    finishedBlocks.sort(key = lambda b: b.index[0])

    dayPaddingStats = DataFrame({'wasteDays':wasteLog}, index=pd.Index(range(1,len(wasteLog)+1), name='binCount'))
    dayPaddingStats['wasteDaysDiff'] = dayPaddingStats.wasteDays.diff()
    dayPaddingStats['wasteDaysPerRealDay'] = dayPaddingStats.wasteDays / totalPlayerDays
    # return wasteFrame 
    blockAssignment = pd.concat(
        (b.assign(block=i, extDayCount=b.index[-1]) 
         for i, b in enumerate(finishedBlocks))
        )
    return blockAssignment, dayPaddingStats



def extendPlayerDaysWithPaddingDays(players : DataFrame, playerDays : DataFrame) -> DataFrame:
    addedDays = players.extDayCount - players.playerDayCount
    addedDays = addedDays[addedDays > 0]
    addedDays = addedDays.map(lambda n: pd.date_range(start='2100-01-01', periods=n, freq='D')).explode()
    addedDays = addedDays.to_frame('day') 
    # addedDays['extDayCount'] = da.players.extDayCount
    addedDays['sdev'] = 1
    addedDays['isRealDay'] = False
    addedDays.set_index('day', append=True, inplace=True)
    # assert len(addedDays.index.intersection(playerDays.index)) == 0

    extPlayerDays = pd.concat([playerDays[['sdev']].assign(isRealDay=True), addedDays], verify_integrity=True).join(players.extDayCount, on='player', validate='m:1')
    extPlayerDays.set_index('extDayCount', append=True, inplace=True)
    extPlayerDays = extPlayerDays.reorder_levels(['extDayCount','player','day'])
    extPlayerDays.sort_index(inplace=True)
    return extPlayerDays