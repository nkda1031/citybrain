import json

from graph_util import Node

class Signal():
    def __init__(self,
             idx,
             intersection,
             outRoadList,
        ):
        self.idx=idx
        self.intersection=intersection        
        self.outRoadDict = {
            'N':outRoadList[0],
            'E':outRoadList[1],
            'S':outRoadList[2],
            'W':outRoadList[3],
        }
        self.inRoadDict = {
            d:road.getOppositRoad() if road is not None else None
                for d,road in self.outRoadDict.items()
        }
    def __str__(self):
        return "Signal(id={},outRoad={})".format(
            self.intersection.interId,
            {k:(v.roadId if v is not None else None) for k,v in self.outRoadDict.items()},
        )
    
    def getRoadSpeedLimitList(self,boundType):
        directionMap={"N":0,"E":1,"S":2,"W":3}
        
        if boundType=="outbound":
            roadDict=self.outRoadDict
        elif boundType=="inbound":
            roadDict=self.inRoadDict
        else:
            raise Exception("not expected")

        speedLimit=[0]*4
        for direction,road in roadDict.items():
            idx=directionMap[direction]
            speedLimit[idx]=road.roadSegment.speedLimit if road is not None else 0
        return speedLimit
            
    def getRelativeRoadLengthList(self,
            baseLength,
            boundType,
        ):
        directionMap={"N":0,"E":1,"S":2,"W":3}
        
        if boundType=="outbound":
            roadDict=self.outRoadDict
        elif boundType=="inbound":
            roadDict=self.inRoadDict
        else:
            raise Exception("not expected")
        
        relativeLength=[0]*4
        for direction,road in roadDict.items():
            idx=directionMap[direction]
            relativeLength[idx]=road.roadSegment.length/baseLength if road is not None else 0
                
        return relativeLength
    
    def solveDirection(self,roadId):
        for d,road in self.outRoadDict.items():
            if road is not None and road.roadId==roadId:
                return d,"out"
        for d,road in self.inRoadDict.items():
            if road is not None and road.roadId==roadId:
                return d,"in"
            
    def getStatDict(self):
        return {
            'interId':self.intersection.interId,
            'numRoadSegment':sum([x is not None for x in self.outRoadDict.values()]),
        }
        
    def getText(self):
        txt = '{} {} {} {} {}\n'.format(
            self.intersection.interId,
            self.outRoadDict['N'].roadId if self.outRoadDict['N'] is not None else -1,
            self.outRoadDict['E'].roadId if self.outRoadDict['E'] is not None else -1,
            self.outRoadDict['S'].roadId if self.outRoadDict['S'] is not None else -1,
            self.outRoadDict['W'].roadId if self.outRoadDict['W'] is not None else -1,
        )
        return txt

    @classmethod
    def fromText(cls,line,idx,interIdToIntersectionDict,roadIdToRoadDict):
        sp=list(map(int,line.rstrip('\n').split(' ')))
        
        def getRoad(roadId):
            return roadIdToRoadDict[roadId] if roadId>=0 else None
        
        return Signal(
            idx,
            interIdToIntersectionDict[sp[0]],
            [
                getRoad(sp[1]),
                getRoad(sp[2]),
                getRoad(sp[3]),
                getRoad(sp[4]),
            ]
        )   
    def exportAsDict(self):
        return {
            'idx':self.idx,
            'interId':self.intersection.interId,
            'outRoadDict':{d:(road.roadId if road is not None else None)for d,road in self.outRoadDict.items()},
            'inRoadDict':{d:(road.roadId if road is not None else None) for d,road in self.inRoadDict.items()},
        }
    
class Intersection():
    def __init__(self, x, y, interId,signal=None):
        self.x = x
        self.y = y
        self.interId = interId
        self.signal = signal
    def __str__(self):
        return "Intersection(id={},signalized={})".format(
            self.interId,
            self.signal is not None,
        )
        
    def setSignal(self,signal):
        self.signal = signal

    def getText(self):
        txt = '{} {} {} {}\n'.format(
            self.y,
            self.x,
            self.interId,
            0 if self.signal is None else 1
        )
        return txt
    
    @classmethod
    def fromText(cls,line):
        sp=line.rstrip('\n').split(' ')
        return Intersection(
            float(sp[1]),
            float(sp[0]),
            int(sp[2]),
        )
    
    def exportAsDict(self):
        return {
            'interId':self.interId,
            'x':self.x,
            'y':self.y,
            'signalized':1 if self.signal is not None else 0,
        }

class IntersectionDataSet():
    def __init__(self,intersectionDict,signalDict):
        self.intersectionDict = intersectionDict
        self.signalDict = signalDict
        
    def printSummary(self):
        print("IntersectionDataSet has {} Intersection, {} Signal.".format(
            len(self.intersectionDict),
            len(self.signalDict),
        ))
        
    def getSignalStat(self):
        return [signal.getStatDict() for signal in self.signalDict.values()]
        
    
    @classmethod
    def fromProcessRoadNet(cls,intersections,agents,roadIdToRoadDict):
        interDSC=IntersectionDataSetCreater()
        interIdToIntersectionDict={}
        for interId,interDict in intersections.items():
            inter = Intersection(
                None,
                None,
                interId
            )
            interIdToIntersectionDict[inter.interId]=inter
            interDSC.intersectionDict[inter.interId]=inter
        for idx,(interId,roadIdList) in enumerate(agents.items()):
            signal = Signal(
                idx,
                interIdToIntersectionDict[interId],
                [(roadIdToRoadDict[roadId] if roadId>=0 else None) for roadId in roadIdList[4:]],
            )
            interIdToIntersectionDict[signal.intersection.interId].setSignal(signal)
        return interDSC.compile()
    
    @classmethod
    def fromText(cls,interLines,signalLines,roadIdToRoadDict):
        interDSC=IntersectionDataSetCreater()
        interIdToIntersectionDict={}
        for line in interLines:
            inter = Intersection.fromText(line)
            interIdToIntersectionDict[inter.interId]=inter
            interDSC.intersectionDict[inter.interId]=inter
        
        for idx,line in enumerate(signalLines):
            signal = Signal.fromText(line,idx,interIdToIntersectionDict,roadIdToRoadDict)
            interIdToIntersectionDict[signal.intersection.interId].setSignal(signal)
        return interDSC.compile()

    def getIntersectionText(self):
        txt = '{}\n'.format(len(self.intersectionDict))
        for inter in self.intersectionDict.values():
            txt += inter.getText()
        return txt
    
    def getSignalText(self):
        txt = '{}\n'.format(len(self.signalDict))
        for signal in self.signalDict.values():
            txt += signal.getText()
        return txt    
    
class IntersectionDataSetCreater():
    def __init__(self):
        self.intersectionDict = {}
        self.idxForSignal=0
    
    def append(self,x, y, interId, outRoadListForSignal):
        inter=Intersection(
            x,
            y,
            interId
        )
        if outRoadListForSignal is not None:
            signal=Signal(self.idxForSignal,inter,outRoadListForSignal)
            self.idxForSignal+=1
            inter.setSignal(signal)
        self.intersectionDict[interId]=inter
    
    def getSignalDict(self):
        return {inter.interId:inter.signal for inter in self.intersectionDict.values() if inter.signal is not None}
    
    def compile(self):
        return IntersectionDataSet(self.intersectionDict,self.getSignalDict())

class PermissibleMovement():
    def __init__(self,
            left,
            through,
            right
        ):
        self.left=left
        self.through=through
        self.right=right
    def getText(self):
        return '{} {} {}'.format(
            1 if self.left else 0,
            1 if self.through else 0,
            1 if self.right else 0,
        )
    @staticmethod
    def fromArray(array):
        return PermissibleMovement(
            array[0]==1,
            array[1]==1,
            array[2]==1,
        )
    def toArray(self):
        return [
            1 if self.left else 0,
            1 if self.through else 0,
            1 if self.right else 0,
        ]
        
class Road():
    def __init__(self,
            roadId,
            roadSegment,
            isInvRoad,
            permissibleMovementList=None
        ):
        if permissibleMovementList is None:
            permissibleMovementList=[
                PermissibleMovement(True,False,False),
                PermissibleMovement(False,True,False),
                PermissibleMovement(False,False,True),
            ]
        self.roadId=roadId
        self.roadSegment=roadSegment
        self.permissibleMovementList=permissibleMovementList
        self.isInvRoad = isInvRoad
        
    def __str__(self):
        return "Road(id={},fromInterID={},toInterID={},len={},move={})".format(
            self.roadId,
            self.getStartInterId(),
            self.getEndInterId(),
            self.roadSegment.length,
            sum([move.toArray()for move in self.permissibleMovementList],[])
        )
    def getOppositRoad(self):
        return self.roadSegment.road if self.isInvRoad else self.roadSegment.roadInv
    def getStartInterId(self):
        return self.roadSegment.toInterId if self.isInvRoad else self.roadSegment.fromInterId
    def getEndInterId(self):
        return self.roadSegment.fromInterId if self.isInvRoad else self.roadSegment.toInterId

    def getNumLane(self,permissibleType=None,permissibleNum=None):
        mapDict={
            'Left':0,
            'Through':1,
            'Right':2,
        }
        if permissibleType is None and permissibleNum is None:
            return len(self.permissibleMovementList)
        elif permissibleType is not None:
            return sum([
                move.toArray()[mapDict[permissibleType]]
                    for move in self.permissibleMovementList
            ])
        elif permissibleNum is not None:
            return sum([
                sum(move.toArray())==permissibleNum
                    for move in self.permissibleMovementList
            ])
        else:
            return sum([
                move.toArray()[mapDict[permissibleType]]==1 and sum(move.toArray())==permissibleNum
                    for move in self.permissibleMovementList
            ])                    
    
    def getStatDict(self):
        return {
            'roadId':self.roadId,
            'permissibleMovementList':str(sum([pm.toArray() for pm in self.permissibleMovementList],[])),
            'numLane':self.getNumLane(),
            'numLaneLeft':self.getNumLane(permissibleType="Left"),
            'numLaneThrough':self.getNumLane(permissibleType="Through"),
            'numLaneRight':self.getNumLane(permissibleType="Right"),
            'numLane1way':self.getNumLane(permissibleNum=1),
            'numLane2way':self.getNumLane(permissibleNum=2),
            'numLane3way':self.getNumLane(permissibleNum=3),
        }
    
    def exportAsDict(self):
        return {
            'roadId':self.roadId,
            'permissibleMovementList':sum([pm.toArray() for pm in self.permissibleMovementList],[]),
            'startInterId':self.getStartInterId(),
            'endInterId':self.getEndInterId(),
            'length':self.roadSegment.length,
            'speedLimit':self.roadSegment.speedLimit,
            'invRoadId':self.getOppositRoad().roadId,
        }
    
class RoadSegment():
    def __init__(self,
            fromInterId,
            toInterId,
            length,
            speedLimit,
            roadId,
            invRoadId,
            permissibleMovementList=None,
            invPermissibleMovementList=None
        ):
        self.fromInterId = fromInterId
        self.toInterId = toInterId
        self.length = length
        self.speedLimit = speedLimit
        
        self.road=Road(roadId,self,False,permissibleMovementList)
        self.roadInv=Road(invRoadId,self,True,invPermissibleMovementList)
        
    def getStatDict(self):
        return {
            'length':self.length,
            'speedLimit':self.speedLimit,
            'minSecToPass':self.length / self.speedLimit,
        }
    
    def getText(self):
        txt = '{} {} {} {} {} {} {} {}\n'.format(
            self.fromInterId,
            self.toInterId,
            self.length,
            self.speedLimit, 
            self.road.getNumLane(),
            self.roadInv.getNumLane(),
            self.road.roadId,
            self.roadInv.roadId,
        )
        txt += ' '.join(map(lambda mov : mov.getText(),self.road.permissibleMovementList))+'\n'
        txt += ' '.join(map(lambda mov : mov.getText(),self.roadInv.permissibleMovementList))+'\n'
        return txt        
    
    @staticmethod
    def _createPermissibleMovementListFromArray(array):
        assert len(array) % 3 ==0
        lanes=int(len(array)/3)
        return [
            PermissibleMovement.fromArray(array[idx*3:idx*3+3])
                for idx in range(lanes)
        ]
    
    @classmethod
    def fromText(cls,lines):
        assert len(lines)==3
        sp0=lines[0].rstrip('\n').split(' ')
        sp1=list(map(int,lines[1].rstrip('\n').split(' ')))
        sp2=list(map(int,lines[2].rstrip('\n').split(' ')))
        return RoadSegment(
            int(sp0[0]),
            int(sp0[1]),
            float(sp0[2]),
            float(sp0[3]),
            int(sp0[6]),
            int(sp0[7]),
            cls._createPermissibleMovementListFromArray(sp1),
            cls._createPermissibleMovementListFromArray(sp2)
        )

class RoadDataSet():
    def __init__(self,roadSegmentList,roadDict,interIdToInboudRoadListDict,interIdToOutboudRoadListDict,interIdToNumRoadSegmentDict):
        self.roadSegmentList = roadSegmentList
        self.roadDict = roadDict
        self.interIdToInboudRoadListDict=interIdToInboudRoadListDict
        self.interIdToOutboudRoadListDict=interIdToOutboudRoadListDict
        self.interIdToNumRoadSegmentDict=interIdToNumRoadSegmentDict
        
    def printSummary(self):
        print("RoadDataSet has {} RoadSegmant, {} Road.".format(
            len(self.roadSegmentList),
            len(self.roadDict),
        ))
    
    def getRoadStat(self):
        return [road.getStatDict() for road in self.roadDict.values()]
    def getRoadSegmentStat(self):
        return [roadSeg.getStatDict() for roadSeg in self.roadSegmentList]
    def getIntersectionStat(self):
        return [{
            'interId':interId,
            'numRoadSegment':NumRoadSeg,
        }for interId,NumRoadSeg in self.interIdToNumRoadSegmentDict.items()] 
        
    def getText(self):
        txt = '{}\n'.format(len(self.roadSegmentList))
        for roadSegment in self.roadSegmentList:
            txt += roadSegment.getText()
        return txt

    def solveRoute(self,interIdList):
        #solve route from interIdList then return roadIdList
        
        def findRoad(roadList,endInterId):
            for road in roadList:
                if road.getEndInterId()==endInterId:
                    return road
            return None
        
        roadList=[]
        for idx in range(len(interIdList)-1):
            fromInterId=interIdList[idx]
            toInterId=interIdList[idx+1]
            road=findRoad(
                self.interIdToOutboudRoadListDict[fromInterId],
                toInterId
            )
            if road is None:
                raise Exception("cannot resolve route")
            roadList.append(road)
        return roadList

    def calcRemaindLengthToGoal(self,vehicle,normalize=False):
        #using route
        road = self.roadDict[vehicle.roadId]
        sumDistance = (road.roadSegment.length - vehicle.distance)
        for roadId in vehicle.route[1:]:
            road = self.roadDict[roadId]
            sumDistance += road.roadSegment.length
            
        if normalize:
            #sumDistance=sumDistance/vehicle.calcOriginalLengthToGoal(self)
            sumDistance=sumDistance/vehicle.minTravelTime
            
        return sumDistance
    
    def calcDelayIndex(self,vehicle):
        #using route
        road = self.roadDict[vehicle.roadId]
        tt_f_r = (road.roadSegment.length - vehicle.distance) / road.roadSegment.speedLimit
        lastTime = vehicle.currentTime if vehicle.goalTime is None else vehicle.goalTime
        
        for roadId in vehicle.route[1:]:
            road = self.roadDict[roadId]
            tt_f_r += road.roadSegment.length / road.roadSegment.speedLimit

        return (lastTime - vehicle.startTime + tt_f_r) / vehicle.minTravelTime    

    @classmethod
    def fromProcessRoadNet(cls,roads):
        def createPermissibleMovementList(roadId,roadDict):
            return [
                PermissibleMovement.fromArray(roadDict['lanes'][laneId])
                    for laneId in range(roadId*100,roadId*100+roadDict['num_lanes'])
            ]
        roadDSC=RoadDataSetCreater()
        alreadyAddedRoadIdList=[]
        for roadId,roadDict in roads.items():
            if roadDict['inverse_road'] in alreadyAddedRoadIdList:
                continue
            alreadyAddedRoadIdList.append(roadId)
            roadSeg=RoadSegment(
                roadDict['start_inter'],
                roadDict['end_inter'],
                roadDict['length'],
                roadDict['speed_limit'],
                roadId,
                roadDict['inverse_road'],
                createPermissibleMovementList(roadId,roadDict),
                createPermissibleMovementList(roadDict['inverse_road'],roads[roadDict['inverse_road']]),
            )
            
            permissibleMovementList=None,
            invPermissibleMovementList=None
            
            roadDSC.roadSegmentList.append(roadSeg)
        return roadDSC.compile()
    
    @classmethod
    def fromText(cls,lines):
        assert len(lines)%3==0
        roadNum=int(len(lines)/3)
        roadDSC=RoadDataSetCreater()
        for idx in range(roadNum):
            roadSeg=RoadSegment.fromText(lines[idx*3:idx*3+3])
            roadDSC.roadSegmentList.append(roadSeg)
        return roadDSC.compile()

    def assertFlow(self,flow):
        roadDict=self.roadDict
        numRoadSegDict=self.interIdToNumRoadSegmentDict
        
        stratRoad=roadDict[flow.roadIdlist[0]]
        startInterId=stratRoad.getStartInterId()
        assert numRoadSegDict[startInterId]<=3, \
            "flow={} is not valid at startRoad={}".format(flow.roadIdlist,str(stratRoad))

        endRoad=roadDict[flow.roadIdlist[-1]]
        endInterId=endRoad.getEndInterId()
        assert numRoadSegDict[endInterId]<=3,  \
            "flow={} is not valid at endRoad={}".format(flow.roadIdlist,str(endRoad))

        for idx in range(len(flow.roadIdlist)-1):
            fromRoad=roadDict[flow.roadIdlist[idx]]
            toRoad=roadDict[flow.roadIdlist[idx+1]]
            assert fromRoad.getEndInterId()==toRoad.getStartInterId(),  \
                "flow={} is not valid btw {} and {}".format(flow.roadIdlist,str(fromRoad),str(toRoad))
    
    def assertFlowDataSet(self,flowDS):
        for flow in flowDS.flowList:
            self.assertFlow(flow)
    
        
class RoadDataSetCreater():
    #roadSegmentId: starts from 0
    #roadId: starts from 1
    #roadId = 2*roadSegmentId+1 or 2*roadSegmentId+2 (for inverted road)
    
    def __init__(self):
        self.roadSegmentList = []
        
    def append(self,
            roadSegmentId,
            fromInterId,
            toInterId,
            length,
            speedLimit,
            permissibleMovementList=None,
            invPermissibleMovementList=None
        ):
        self.roadSegmentList.append(RoadSegment(
            fromInterId,
            toInterId,
            length,
            speedLimit,
            self.getRoadId(roadSegmentId,False),
            self.getRoadId(roadSegmentId,True),
            permissibleMovementList,
            invPermissibleMovementList
        ))
    
    def compile(self):
        return RoadDataSet(
            self.roadSegmentList,
            self.getRoadDict(),
            self.getInterIdToInboudRoadListDict(),
            self.getInterIdToOutboudRoadListDict(),
            self.getInterIdToNumRoadSegmentDict(),
        )
    
    @staticmethod
    def getRoadId(roadSegmentId,inv=False):
        return 2*roadSegmentId+(2 if inv else 1)
    
    def getRoad(self,roadSegmentId,inv=False):
        roadSegment=self.roadSegmentList[roadSegmentId]
        return roadSegment.roadInv if inv else roadSegment.road
    
    def getRoadDict(self):
        roadIdToRoadDict={}
        for roadSeg in self.roadSegmentList:
            roadIdToRoadDict[roadSeg.road.roadId]=roadSeg.road
            roadIdToRoadDict[roadSeg.roadInv.roadId]=roadSeg.roadInv
        return roadIdToRoadDict
    
    def getInterIdToInboudRoadListDict(self):
        dic={}
        for road in self.getRoadDict().values():
            interId=road.getEndInterId()
            if interId not in dic:
                dic[interId]=[]
            dic[interId].append(road)
        return dic    
    
    def getInterIdToOutboudRoadListDict(self):
        dic={}
        for road in self.getRoadDict().values():
            interId=road.getStartInterId()
            if interId not in dic:
                dic[interId]=[]
            dic[interId].append(road)
        return dic    

    def getInterIdToNumRoadSegmentDict(self):
        dic={}
        for roadSeg in self.roadSegmentList:
            if roadSeg.fromInterId not in dic:
                dic[roadSeg.fromInterId]=0
            dic[roadSeg.fromInterId]+=1
            if roadSeg.toInterId not in dic:
                dic[roadSeg.toInterId]=0
            dic[roadSeg.toInterId]+=1
        return dic    
    
laneMapDict={
    'N':['E','S','W'],
    'E':['S','W','N'],
    'S':['W','N','E'],
    'W':['N','E','S'],
}    
    
class RoadNet():
    def __init__(self,roadDataSet,intersectionDataSet):
        self.roadDataSet=roadDataSet
        self.intersectionDataSet=intersectionDataSet
        
    def getNextRoadId(self,roadId,laneId):
        #roadIDの行き先が信号のない交差点である場合はNoneを返します。
        #roadIDの行き先の交差点において、レーンが示す先の方角にroadがない場合はNoneを返します。
        endInterId = self.roadDataSet.roadDict[roadId].getEndInterId()
        if endInterId in self.intersectionDataSet.signalDict:
            signal = self.intersectionDataSet.signalDict[endInterId]
            inDirection,_=signal.solveDirection(roadId)
            outDirection=laneMapDict[inDirection][laneId]
            road=signal.outRoadDict[outDirection]
            if road is None:
                return None
            else:
                return road.roadId
        else:
            return None
    
    def printSummary(self):
        self.roadDataSet.printSummary()
        self.intersectionDataSet.printSummary()
    
    def getText(self):
        txt = ''
        txt += self.intersectionDataSet.getIntersectionText()
        txt += self.roadDataSet.getText()
        txt += self.intersectionDataSet.getSignalText()
        return txt
    def saveText(self,out_path):
        with open(out_path,"w") as f:
            f.write(self.getText())
    
    def exportAsJson(self,out_path):
        def write(dic):
            json.dump(dic, open(out_path, 'w'), indent=2)
        write(
            {
                'intersections':{interId:inter.exportAsDict() for interId,inter in self.intersectionDataSet.intersectionDict.items()},
                'signals':{interId:signal.exportAsDict() for interId,signal in self.intersectionDataSet.signalDict.items()},
                'roads':{roadId:road.exportAsDict() for roadId,road in self.roadDataSet.roadDict.items()},
            }
        )
        
    @staticmethod
    def createFromProcessRoadNet(intersections,roads,agents):
        roadDS=RoadDataSet.fromProcessRoadNet(roads)
        interDS=IntersectionDataSet.fromProcessRoadNet(intersections,agents,roadDS.roadDict)
        return RoadNet(roadDS,interDS)
        
    @staticmethod
    def createFromRoadNetFile(path):
        with open(path, 'r') as f:
            lines = f.readlines()
            
        interNum=int(lines[0].rstrip('\n'))
        interLines=lines[1:1+interNum]
        roadNum=int(lines[1+interNum].rstrip('\n'))
        roadLines=lines[2+interNum:2+interNum+roadNum*3]
        signalNum=int(lines[2+interNum+roadNum*3].rstrip('\n'))
        signalLines=lines[3+interNum+roadNum*3:3+interNum+roadNum*3+signalNum]
        
        roadDS=RoadDataSet.fromText(roadLines)
        interDS=IntersectionDataSet.fromText(interLines,signalLines,roadDS.roadDict)
            
        return RoadNet(roadDS,interDS)
 
    @staticmethod
    def createFromGraph(graph,signalizedNodeIdList=[],speedLimit=20,onEarth=False):
        roadDSC=RoadDataSetCreater()
        for link in graph.linkList:
            roadDSC.append(
                link.linkId,
                link.node1.nodeId,
                link.node2.nodeId,
                Node.calcDistance(link.node1,link.node2,onEarth=onEarth),
                link.speedLimit if link.speedLimit is not None else speedLimit,
            )
        
        def getRoad(node,key):
            idx=node.checkLink(key)
            if idx==0:
                return None
            else:
                return roadDSC.getRoad(node.linkDict[key].linkId,idx==2)
        roadDS=roadDSC.compile()
                
        interDSC=IntersectionDataSetCreater()
        for node in graph.nodeList:
            outRoadList = [
                getRoad(node,'N'),
                getRoad(node,'E'),
                getRoad(node,'S'),
                getRoad(node,'W'),
            ] if node.nodeId in signalizedNodeIdList else None
            interDSC.append(node.x,node.y,node.nodeId,outRoadList)
        interDS=interDSC.compile()
            
        #assertion
        for signal in interDS.signalDict.values():
            for road in signal.outRoadDict.values():
                if road is not None:
                    assert road in roadDS.interIdToOutboudRoadListDict[signal.intersection.interId], "{} not in {}".format(road,roadDS.interIdToOutboudRoadListDict[signal.intersection.interId])

        return RoadNet(roadDS,interDS)