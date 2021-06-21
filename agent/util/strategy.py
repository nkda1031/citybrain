import random
import json

from road_tracer import RoadTracer
from signal_state_util import SignalState,LaneVehicleNumCalc
from dnn_model import RunDistanceModel,SavedRunDistanceModel
from simulation import _BaseActionSolver
from strategy_runner import StrategyRunner

class StrategyActionSolver(_BaseActionSolver):
    def __init__(self,
            strategy = None,
            export_agent_log_path=None,
            action_test_count=1
        ):
        super().__init__()
        self.action_test_count=action_test_count # recursively calculate actions based on asuumed actions
        
        if strategy is None:
            self.strategy=Strategy.getDefaultStrategy()
        else:
            self.strategy=strategy
        self.strategyRunner=StrategyRunner(self.strategy)
            
        self.export_agent_log_path=export_agent_log_path
        if self.export_agent_log_path is not None:
            with open(self.export_agent_log_path,"w") as f:
                pass #ファイルの中身を空にする
            
        self.actionLogDict=None

    def setRoadNet(self,roadNet):
        self.roadNet=roadNet
        self.strategy.setRoadNet(roadNet)
        
    def startFirstEpisode(self,signalizedInterIdList):
        self.strategy.setWorldSignalState(self.worldSignalState)
        self.actionLogDict={}
    def startFollowingEpisode(self,signalizedInterIdList):
        self.strategy.setWorldSignalState(self.worldSignalState)
        self.actionLogDict={}
    def exportActionLog(self,out_path):
        with open(out_path, 'w') as f:
            json.dump(self.actionLogDict, f, indent=2)
    
    def getStatMessage(self):
        return "calcReward:{}".format(self.strategyRunner.statForCalcRewardTime.getString())
    
    def getExtraMessage(self):
        return self.strategy.getName()
    
    def decideActions(self,
            observations,
            prevActCountInEpisode,
            runType="eval",
            exitType=False,
            debug=False
        ):
        
        chooseFromMultipleBestPhases=False

        actions=None
        for i in range(self.action_test_count):
            actions_temp=self.strategyRunner.getActionsFunc(
                observations,
                chooseFromMultipleBestPhases=chooseFromMultipleBestPhases,
                debug=debug
            )
            
            if actions is not None:
                if actions_temp==actions:
                    break
            actions=actions_temp
        
        for interId,bestPhase in actions.items():
            signalState=self.worldSignalState.signalStateDict[interId]            
            if bestPhase != signalState.phase:
                signalState.changePhase(bestPhase,observations.current_time,debug=debug)
            else:
                signalState.keepPhase(observations.current_time)
        
        self.actionLogDict[observations.current_time]=actions
            
        if self.export_agent_log_path is not None:
            with open(self.export_agent_log_path,"a") as f:
                for interId in actions:
                    signalState=self.worldSignalState.signalStateDict[interId]
                    f.write("{} {} {} {}\n".format(
                        observations.current_time,
                        interId,
                        signalState.phase,
                        signalState.phaseToRewardDict,
                    ))
            
        return actions    

class Strategy():
    def __init__(self):
        raise "this is static class"
    @classmethod
    def getDefaultStrategy(cls):
        return cls.createStrategy("run_distance")
        
    @staticmethod
    def createStrategy(strategy_name,options_dict={}):
        if strategy_name=="num_vehicle":
            return NumVehicleStrategy(**options_dict)
        elif strategy_name=="run_distance":
            return RunDistanceStrategy(**options_dict)
        elif strategy_name=="dnn_run_distance":
            return DnnRunDistanceStrategy(**options_dict)
        elif strategy_name=="dnn_saved_run_distance":
            return DnnSavedRunDistanceStrategy(**options_dict)
        else:
            raise Exception("not supported strategy")    
    
class _BaseStrategy():
    def __init__(self):
        self.roadNet=None
        self.worldSignalState=None
    
    def setRoadNet(self,roadNet):
        self.roadNet=roadNet
        
    def setWorldSignalState(self,worldSignalState):
        self.worldSignalState=worldSignalState

    def calcReward(self,observations,interId,debug=False):
        raise Exception("please override function calcReward")
        
    def calcRewardAll(self,observations,debug=False):
        rewardDict={}
        for interId in self.worldSignalState.signalStateDict:
            rewardDict[interId]=self.calcReward(observations,interId,debug=debug)
        return rewardDict
        
class RandomMixStrategy():
    # 時間step毎に確率的にstrategyを切り替える複合的なstrategy
    # ※ある時間に使用するstrategyは固定であり、inter毎の切り替えはサポートしない
    def __init__(self,
            strategyDict
        ):
        self.roadNet=None
        self.worldSignalState=None
        
        self.batchCalcOnly=True
        self.strategyDict=strategyDict
        self.lastSelectedStrategy=None
        
    def setRoadNet(self,roadNet):
        self.roadNet=roadNet
        for strategy in self.strategyDict:
            strategy.setRoadNet(roadNet)
        
    def setWorldSignalState(self,worldSignalState):
        #worldSignalStateはstrategy間で共有する
        self.worldSignalState=worldSignalState
        for strategy in self.strategyDict:
            strategy.setWorldSignalState(worldSignalState)
        
    def getName(self):
        #calcRewardAllを呼んだ後にgetNameをすると、alcRewardAll時に使用したstrategyの名前を得る
        return self.lastSelectedStrategy.getName()
        
    def _selectStrategy(self):
        totalWeight=sum(self.strategyDict.values())
        select=random.random()*totalWeight
        
        sumWeight=0
        for strategy,weight in self.strategyDict.items():
            sumWeight+=weight
            if select<sumWeight:
                return strategy
        return self.strategyDict[-1]
        
    def calcRewardAll(self,observations,debug=False):
        strategy = self._selectStrategy()
        self.lastSelectedStrategy = strategy
        return strategy.calcRewardAll(observations,debug=debug)
    
class MainSubStrategy():
    # main strategyで判断に迷う（１位と２位のスコアが近い）場合にsub strategyを使用する複合的なstrategy
    def __init__(self,
            mainStrategy,
            subStrategy,
        ):
        self.roadNet=None
        self.worldSignalState=None
        
        self.batchCalcOnly=True
        self.mainStrategy=mainStrategy
        self.subStrategy=subStrategy
        
    def setRoadNet(self,roadNet):
        self.roadNet=roadNet
        self.mainStrategy.setRoadNet(roadNet)
        self.subStrategy.setRoadNet(roadNet)
        
    def setWorldSignalState(self,worldSignalState):
        #worldSignalStateはstrategy間で共有する
        self.worldSignalState=worldSignalState
        self.mainStrategy.setWorldSignalState(worldSignalState)
        self.subStrategy.setWorldSignalState(worldSignalState)
        
    def getName(self):
        return mainStrategy.getName()+"&"+subStrategy.getName()
        
    def calcRewardAll(self,observations,debug=False):
        mainRewardAll=self.mainStrategy.calcRewardAll(observations,debug=debug)
        subRewardAll=self.subStrategy.calcRewardAll(observations,debug=debug)
        
        confidentRatio=1.2
        mixedRewardAll={}
        _mainCnt=0
        _subCnt=0
        for interId,rewards in mainRewardAll.items():
            sortedRewards=sorted(rewards.values(),reverse=True)
            if sortedRewards[1]>0 and sortedRewards[0]/sortedRewards[1]<confidentRatio:
                mixedRewardAll[interId]=subRewardAll[interId]
                _subCnt+=1
            else:
                mixedRewardAll[interId]=mainRewardAll[interId]
                _mainCnt+=1
        print("[mix strategy] main:{},sub:{}".format(_mainCnt,_subCnt))
        return mixedRewardAll

class DnnSavedRunDistanceStrategy(_BaseStrategy):
    def __init__(self,
            savedModelPath,
            numSegmentInbound=9,
            numSegmentOutbound=9,
            segmentLength=25,
        ):
        super().__init__()
        self.batchCalcOnly=True
        self.model=SavedRunDistanceModel(
            savedModelPath,
            numSegmentInbound,
            numSegmentOutbound,
            segmentLength,
        )

    def getName(self):
        return "dnn_saved_run_distance"
    
    def calcRewardAll(self,observations,debug=False):
        tracer=RoadTracer(
            self.worldSignalState,
            observations.vehicleDS,
            self.roadNet.intersectionDataSet.signalDict,
            self.roadNet.roadDataSet,
        )
        probs,interIds=self.model.calcPhaseProbAll(tracer)
        
        return {
            int(interId) : {
                (idx+1):p
                    for idx,p in enumerate(prob)
            }
            for interId,prob in zip(interIds,probs)
        }
    
class DnnRunDistanceStrategy(_BaseStrategy):
    def __init__(self,
            checkpointPath,
            numSegmentInbound=9,
            numSegmentOutbound=9,
            segmentLength=25,
            unitsDict={},
            dimVehiclesVec=32,
            version=1,
        ):
        super().__init__()
        self.batchCalcOnly=True
        self.segmentLength=segmentLength
        
        self.model=RunDistanceModel(
                numSegmentInbound,
                numSegmentOutbound,
                unitsDict=unitsDict,
                dimVehiclesVec=dimVehiclesVec,
                version=version
            )
        self.model.load_weights(checkpointPath)
        
    def getName(self):
        return "dnn_run_distance"
    
    def calcRewardAll(self,observations,debug=False):
        tracer=RoadTracer(
            self.worldSignalState,
            observations.vehicleDS,
            self.roadNet.intersectionDataSet.signalDict,
            self.roadNet.roadDataSet,
        )
        _,probs,interIds = self.model.calcEmbeddingVecAll(tracer,self.segmentLength)
        return {
            int(interId) : {
                (idx+1):p
                    for idx,p in enumerate(prob)
            }
            for interId,prob in zip(interIds,probs)
        }

class RunDistanceStrategy(_BaseStrategy):
    def __init__(self,
            timeThresForCalcReward=10.2,
            prohibitDecreasingGoalDistance=True,
            prohibitDecreasingSpeed=True,
            
        ):
        super().__init__()
        self.batchCalcOnly=False
        self.timeThresForCalcReward=timeThresForCalcReward
        self.prohibitDecreasingGoalDistance=prohibitDecreasingGoalDistance
        self.prohibitDecreasingSpeed=prohibitDecreasingSpeed
    
    def getName(self):
        return "run_distance"
    
    def calcReward(self,observations,interId,debug=False):
        tracer=RoadTracer(
            self.worldSignalState,
            observations.vehicleDS,
            self.roadNet.intersectionDataSet.signalDict,
            self.roadNet.roadDataSet,
        )
        reward=tracer.createPhaseToRunDistanceDict(
            interId,
            self.timeThresForCalcReward,
            prohibitDecreasingGoalDistance=self.prohibitDecreasingGoalDistance,
            prohibitDecreasingSpeed=self.prohibitDecreasingSpeed,
            debug=debug
        )
        
        return {
            phase:round(x,1)
                for phase,x in reward.items()
        } #ランダム要素をなくす為に1桁に丸める