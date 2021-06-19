import numpy as np
import copy

class SignalState():
    def __init__(self,interId):
        self.interId=interId
        self.phase=1
        self.tempPhase=1
        self.lastChangeTime=0
        self.phaseToRewardDict=None
        self.phaseToNoneZeroRewardTimeCountDict={}
        
        self.prevPhase=1
        self.prevTimeStepForDQN=None
        self.prevPolicyStepForDQN=None
    def setPreviousStepInfo(self,
        prevTimeStepForDQN=None,
        prevPolicyStepForDQN=None,
    ):
        if prevTimeStepForDQN is not None:
            self.prevTimeStepForDQN=prevTimeStepForDQN
        if prevPolicyStepForDQN is not None:
            self.prevPolicyStepForDQN=prevPolicyStepForDQN
        
    def record(self,phaseToRewardDict):
        self.phaseToRewardDict=phaseToRewardDict
        for phase,reward in phaseToRewardDict.items():
            if reward != 0:
                if phase not in self.phaseToNoneZeroRewardTimeCountDict:
                    self.phaseToNoneZeroRewardTimeCountDict[phase] = 0
                self.phaseToNoneZeroRewardTimeCountDict[phase] += 1
            else:
                self.phaseToNoneZeroRewardTimeCountDict[phase] = 0
    def changePhase(self,phase,current_time,debug=False):
        if debug:
            print("[{}] change signal {} phase {} to {}".format(
                current_time,
                self.interId,
                self.phase,
                phase)
            )
        self.prevPhase = self.phase
        self.phase = phase
        self.tempPhase = phase
        self.lastChangeTime = current_time
    def keepPhase(self,current_time,debug=False):
        if debug:
            print("[{}] keep signal {} phase {}".format(
                current_time,
                self.interId,
                self.phase,
            ))
        self.prevPhase = self.phase
        self.tempPhase = self.phase
    def changeTempPhase(self,phase):
        self.tempPhase = phase
        
    def getFilteredPhaseToNoneZeroRewardTimeCountDict(self,targetPhaseList):
        return {
            phase : notMinRewardTimeCount
                for phase,notMinRewardTimeCount in self.phaseToNoneZeroRewardTimeCountDict.items()
                    if phase in targetPhaseList
        }    
    def getPassableDirectionList(self,useTempPhase=True):
        return self.phaseToPassableDirectionListDict[self.tempPhase if useTempPhase else self.phase]

    def getPassableEncoding(self,useTempPhase=True):
        directionList=self.getPassableDirectionList(useTempPhase)
        return [(1 if direction in directionList else 0) for direction in self.allDirectionList]

    def getPassableEncodingWithoutRightTurn(self,useTempPhase=True):
        directionList=self.getPassableDirectionList(useTempPhase)
        return [(1 if direction in directionList else 0) for direction in self.withoutRightTurnDirectionList]
        
    def isPassableDirections(self,currentDirection,nextDirection):
        return (currentDirection,nextDirection) in self.getPassableDirectionList()
    
    def isPassableRoads(self,currentRoadId,nextRoadId,signalDict):
        signal = signalDict[self.interId]
        currentDirection,currentType=signal.solveDirection(currentRoadId)
        assert currentType=="in"
        nextDirection,nextType=signal.solveDirection(nextRoadId)
        assert nextType=="out"
        return self.isPassableDirections(currentDirection,nextDirection)

    def getPassablePreviousRoadList(self,roadId,signalDict):
        signal = signalDict[self.interId]
        current_direction,_ = signal.solveDirection(roadId)
        return [
            prev_road
                for prev_direction,prev_road in signal.inRoadDict.items()
                    if prev_road is not None and self.isPassableDirections(prev_direction,current_direction)
        ]
    
    # mapping a phase_id to passable directions
    # USAGE:
    #     for from_direction,to_direction in phaseToPassableDirectionListDict[phase_id]:
    phaseToPassableDirectionListDictExceptRightTurn={
        1:[('N','E'),('S','W')],
        2:[('N','S'),('S','N')],
        3:[('E','S'),('W','N')],
        4:[('E','W'),('W','E')],
        5:[('N','E'),('N','S')],
        6:[('E','S'),('E','W')],
        7:[('S','W'),('S','N')],
        8:[('W','N'),('W','E')],
    }
    rightTurnList=[('N','W'),('W','S'),('S','E'),('E','N')]
    phaseToPassableDirectionListDict={
        1:[('N','E'),('S','W')]+rightTurnList,
        2:[('N','S'),('S','N')]+rightTurnList,
        3:[('E','S'),('W','N')]+rightTurnList,
        4:[('E','W'),('W','E')]+rightTurnList,
        5:[('N','E'),('N','S')]+rightTurnList,
        6:[('E','S'),('E','W')]+rightTurnList,
        7:[('S','W'),('S','N')]+rightTurnList,
        8:[('W','N'),('W','E')]+rightTurnList,
    }
    
    phaseToStopDirectionListDict={
        k:list(set([('N','E'),('N','S'),('N','W'),('E','S'),('E','W'),('E','N'),('S','W'),('S','N'),('S','E'),('W','N'),('W','E'),('W','S')])-set(v))
            for k,v in phaseToPassableDirectionListDict.items()
    }
    allDirectionList=[
        ('N','E'),('N','S'),('N','W'),
        ('E','S'),('E','W'),('E','N'),
        ('S','W'),('S','N'),('S','E'),
        ('W','N'),('W','E'),('W','S'),
    ]
    withoutRightTurnDirectionList=[
        ('N','E'),('N','S'),
        ('E','S'),('E','W'),
        ('S','W'),('S','N'),
        ('W','N'),('W','E'),
    ]
    
class WorldSignalState():
    def __init__(self):
        self.signalStateDict={}
    def clone(self):
        cloned=WorldSignalState()
        for signalState in  self.signalStateDict.values():
            cloned.add(copy.deepcopy(signalState))
        return cloned
    def add(self,signalState):
        self.signalStateDict[signalState.interId]=signalState
    def isPassableRoads(self,interId,currentRoadId,nextRoadId,signalDict):
        if interId in self.signalStateDict: #signalized inter
            return self.signalStateDict[interId].isPassableRoads(currentRoadId,nextRoadId,signalDict)
        else:
            return True
    def getPassablePreviousRoadList(self,road,signalDict):
        startInterId=road.getStartInterId()
        if startInterId in self.signalStateDict: #signalized inter
            return self.signalStateDict[startInterId].getPassablePreviousRoadList(road.roadId,signalDict)
        else:
            raise Exception("not expected")

class LaneVehicleNumCalc():
    @classmethod
    def createPhaseToNumVehicelsDictFromNumVehicleOnLane(cls,
            lane_vehicle_num,
            phaseToDirectionListDict,
            limit_4_signal_only=False,
        ):
        # count of vehicles for each signal phase
        v=np.array(lane_vehicle_num[1:])
        v=np.maximum(v,0)
        return {
            phaseId: sum([
                        v[cls.directionToLane12indexDict[from_d][to_d]-1]
                            for from_d,to_d in phaseToDirectionListDict[phaseId]
                    ])
                for phaseId in range(1,5 if limit_4_signal_only else 9)
        }
    
    # mapping a lane to index for "lane_vehicle_num" list
    # USAGE:
    #     laneIndex = directionToLane12indexDict[current_direction][next_direction]
    #
    # current_direction : current road direction. one of 'N', 'E', 'S' or 'W'
    # next_direction : next road direction. one of 'N', 'E', 'S' or 'W'
    directionToLane12indexDict={
        'N':{
            'E':1,
            'S':2,
            'W':3,
        },
        'E':{
            'S':4,
            'W':5,
            'N':6,
        },
        'S':{
            'W':7,
            'N':8,
            'E':9,
        },
        'W':{
            'N':10,
            'E':11,
            'S':12,
        },
    }