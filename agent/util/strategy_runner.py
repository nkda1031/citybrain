import time

class Stat:
    def __init__(self):
        self.sum=0.
        self.sum2=0.
        self.n=0
    def add(self,val):
        self.n+=1
        self.sum+=val
        self.sum2+=val*val
    def getAverage(self):
        return self.sum/self.n
    def getStddev(self):
        u=self.getAverage()
        return self.sum2/self.n-u*u
    def getString(self):
        return "{:.3g} +- {:.3g} ({} samples)".format(
            self.getAverage(),
            self.getStddev(),
            self.n,
        )
    
class StrategyRunner():
    def __init__(self,strategy):
        self.strategy=strategy
        self.getActionsFunc=self._getActionsBatch if strategy.batchCalcOnly else self._getActionsOneByOne
        self.statForCalcRewardTime=Stat()

    @staticmethod
    def _chooseKeyListOfMaxValue(targetDict):
        maxVal=max(targetDict.values())
        return [key for key,val in targetDict.items() if val==maxVal]
        
    def _getActionsBatch(self,
            observations,
            chooseFromMultipleBestPhases=False,
            debug=False
        ):
        # get actions
        
        t0=time.time()
        interIdToPhaseToRewardDictDict=self.strategy.calcRewardAll(observations,debug=debug)
        t1=time.time()
        self.statForCalcRewardTime.add(t1-t0)
            
        actions = {}
        for interId,phaseToRewardDict in interIdToPhaseToRewardDictDict.items():
            signalState=self.strategy.worldSignalState.signalStateDict[interId]
            signalState.record(phaseToRewardDict)

            bestPhaseList = self._chooseKeyListOfMaxValue(phaseToRewardDict)
            if signalState.phase in bestPhaseList:
                # keep signal phase if current phase is one of the best options
                signalState.changeTempPhase(signalState.phase)
                continue              

            if len(bestPhaseList)>1 and chooseFromMultipleBestPhases:
                bestPhaseList = self._chooseKeyListOfMaxValue(
                    signalState.getFilteredPhaseToNoneZeroRewardTimeCountDict(bestPhaseList)
                )
            bestPhase=bestPhaseList[0]
            actions[interId] = bestPhase
            
            ############# change phase of this signal ################
            signalState.changeTempPhase(bestPhase)
                
        return actions
    
    def _getActionsOneByOne(self,
            observations,
            chooseFromMultipleBestPhases=False,
            debug=False
        ):
        # get actions
        actions = {}
        for interId in self.strategy.worldSignalState.signalStateDict:
            signalState=self.strategy.worldSignalState.signalStateDict[interId]

            t0=time.time()
            phaseToRewardDict=self.strategy.calcReward(observations,interId,debug=debug)
            t1=time.time()
            self.statForCalcRewardTime.add(t1-t0)
            
            signalState.record(phaseToRewardDict)

            bestPhaseList = self._chooseKeyListOfMaxValue(phaseToRewardDict)
            if signalState.phase in bestPhaseList:
                # keep signal phase if current phase is one of the best options
                signalState.changeTempPhase(signalState.phase)
                continue              

            if len(bestPhaseList)>1 and chooseFromMultipleBestPhases:
                bestPhaseList = self._chooseKeyListOfMaxValue(
                    signalState.getFilteredPhaseToNoneZeroRewardTimeCountDict(bestPhaseList)
                )
            bestPhase=bestPhaseList[0]
            actions[interId] = bestPhase
            
            ############# change phase of this signal ################
            signalState.changeTempPhase(bestPhase)
                
        return actions
