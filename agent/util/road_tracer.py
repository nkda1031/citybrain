import math

from signal_state_util import SignalState,LaneVehicleNumCalc

VEHICLE_MERGIN_METER=5
PHASE_CHANGE_STOP_SEC=6
#PHASE_CHANGE_STOP_SEC=5
SPEED_LIMIT_IN_SLOW_VEHICLES=5

def interpolate(x1,x2,w1,w2):
    return (x1*w1+x2*w2)/(w1+w2)

class RoadTracer():
    #iterateDistanceSegmentedVehicleList
    # -> iterateDistanceTargetVehicleList
    #calcSumRunDistance
    # -> iterateDistanceTargetVehicleList
    
    def __init__(self,worldSignalState,vehicleDS,signalDict,roadDS):
        self.worldSignalState=worldSignalState
        self.vehicleDS=vehicleDS
        self.signalDict=signalDict
        self.roadDS=roadDS

    def calcSumRunDistance(self,
            vehicleDSAfterRun,
            interId,
            distanceFromInter,
            normalize=False
        ):
        #using route
        sumRunDistance=0
        for vehicleList,_ in self.iterateDistanceTargetVehicleList(
                interId,
                0,
                distanceFromInter,
                "inbound",
            ):
            for vehicle in vehicleList:
                if vehicle.vehicleId in vehicleDSAfterRun.existingVehicleDict:
                    futureVehicle = vehicleDSAfterRun.existingVehicleDict[vehicle.vehicleId]
                    sumRunDistance+=futureVehicle.calcRunDistance(vehicle,self.roadDS,normalize=normalize)
                else:
                    sumRunDistance+=vehicle.calcRemaindLengthToGoal(self.roadDS,normalize=normalize)
        return sumRunDistance
    
    def getRoadSpeedLimitList(self,interId,boundType):
        return self.signalDict[interId].getRoadSpeedLimitList(
            boundType,
        )
    
    def getRelativeRoadLengthList(self,interId,segmentLength,boundType):
        return self.signalDict[interId].getRelativeRoadLengthList(
            segmentLength,
            boundType,
        )
    def connectedInterSignalized(self,interId):
        signal = self.signalDict[interId]
        directionMap={"N":0,"E":1,"S":2,"W":3}
        
        signalized=[-1]*4
        for direction,road in signal.inRoadDict.items():
            if road is not None:
                idx=directionMap[direction]
                signalized[idx]=1 if road.getStartInterId() in self.signalDict else 0
        return signalized
        
    def _createDistanceTargetOutboundVehicleList(self,
            minDistanceFromInter,
            maxDistanceFromInter,
            roadId,
        ):
        #roadIdを走行している、
        #かつroadIdの始点から測ってminDistanceFromInterからmaxDistanceFromInterの距離に存在するvehicleを返します。
        #minDistanceFromInterがroadIdのlengthを超える場合、常の空のリストを返します
        road = self.roadDS.roadDict[roadId]
        roadLen=road.roadSegment.length
        targetVehicleOnRoadList=[]
        for vehicle in reversed(self.vehicleDS.getSortedVehicleList(roadId)):
            distFromInter=vehicle.distance
            if distFromInter>=maxDistanceFromInter:
                break
            elif distFromInter<minDistanceFromInter:
                continue
            targetVehicleOnRoadList.append(vehicle)
        return targetVehicleOnRoadList
    def _createDistanceTargetInboundVehicleList(self,
            minDistanceFromInter,
            maxDistanceFromInter,
            roadId,
            nextRoadId
        ):
        #現在roadIdにいて、nextRoadIdに進む道路を走行しており、
        #かつroadIdの終点から測ってminDistanceFromInterからmaxDistanceFromInterの距離に存在するvehicleを返します。
        #maxDistanceFromInterがroadIdのlengthを超える場合でもroadId上の車両以外を含めません
        
        road = self.roadDS.roadDict[roadId]
        roadLen=road.roadSegment.length
        targetVehicleOnRoadList=[]
        for vehicle in self.vehicleDS.getSortedVehicleList(roadId,nextRoadId):
            distFromInter=roadLen-vehicle.distance
            if distFromInter>=maxDistanceFromInter:
                break
            elif distFromInter<minDistanceFromInter:
                continue
            targetVehicleOnRoadList.append(vehicle)
            
        return targetVehicleOnRoadList
    
    @staticmethod
    def _sumDictValues(keyList,targetDict):
        summed=0
        for key in keyList:
            if key in targetDict:
                summed+=targetDict[key]
        return summed
    
    def calcVehicleNumAndSpeedOnSegmentedOutboundRoad(self,
            interId,
            numSegment,
            segmentLength,
        ):
        directionMap={"N":0,"E":1,"S":2,"W":3}
        
        vehicleNumList=[0]*(4*numSegment)
        vehicleSpeedList=[0]*(4*numSegment)
        for seg,current_direction,vehicleList in self.iterateDistanceSegmentedVehicleList(
                interId,
                numSegment,
                segmentLength,
                boundType="outbound"
            ):
            idx=directionMap[current_direction]
            vehicleNumList[idx+seg*4]=len(vehicleList)
            vehicleSpeedList[idx+seg*4]=sum([v.speed for v in vehicleList])
        return vehicleNumList,vehicleSpeedList
    
    def calcVehicleNumAndSpeedOnSegmentedInboundLane(self,
            interId,
            numSegment,
            segmentLength,
        ):
        vehicleNumList=[0]*(12*numSegment)
        vehicleSpeedList=[0]*(12*numSegment)
        for seg,(current_direction,next_direction),vehicleList in self.iterateDistanceSegmentedVehicleList(
                interId,
                numSegment,
                segmentLength,
                boundType="inbound",
            ):
            idx=LaneVehicleNumCalc.directionToLane12indexDict[current_direction][next_direction]-1
            vehicleNumList[idx+seg*12]=len(vehicleList)
            vehicleSpeedList[idx+seg*12]=sum([v.speed for v in vehicleList])
        return vehicleNumList,vehicleSpeedList
    
    def iterateDistanceSegmentedVehicleList(self,
            interId,
            numSegment,
            segmentLength,
            boundType="inbound",
        ):
        for seg in range(numSegment):
            minDistFromInter=seg*segmentLength
            maxDistFromInter=(seg+1)*segmentLength
            for vehicleList,directions in self.iterateDistanceTargetVehicleList(
                    interId,
                    minDistFromInter,
                    maxDistFromInter,
                    boundType,
                ):
                yield seg,directions,vehicleList
    
    def iterateDistanceTargetVehicleList(self,
            interId,
            minDistanceFromInter,
            maxDistanceFromInter,
            boundType="inbound",
        ):
        #boundTypeが"inbound"の場合、下記のtupleをイテレータします。
        # tupleの要素0：下記の進行方向を満たし、かつ指定条件（※１）を満たすvehicleのList
        # tupleの要素1：交差点での進行前後の方向を表すtuple（current_direction,next_direction）、currentから侵入しnextに抜けることを表す
        #
        #指定条件（※１）
        #・interIdからの距離がminDistanceFromInter以上でmaxDistanceFromInterより小さいこと
        #・interIdに直接接続しているinboundな道路上にいること
        #
        #boundTypeが"outbound"の場合、下記のtupleをイテレータします。
        # tupleの要素0：下記の進行方向を満たし、かつ指定条件（※１）を満たすvehicleのList
        # tupleの要素1：交差点での進行後の方向を表すcurrent_direction
        #
        #指定条件（※１）
        #・interIdからの距離がminDistanceFromInter以上でmaxDistanceFromInterより小さいこと
        #・interIdに直接接続しているoutboundな道路上にいること
        signal = self.signalDict[interId]
        if boundType=="inbound":
            for current_direction,road in signal.inRoadDict.items():
                for next_direction,nextRoad in signal.outRoadDict.items():
                    if road is None or nextRoad is None or next_direction==current_direction:
                        continue
                    targetVehicleList=self._createDistanceTargetInboundVehicleList(
                        minDistanceFromInter,
                        maxDistanceFromInter,
                        road.roadId,
                        nextRoad.roadId,
                    )
                    yield targetVehicleList,(current_direction,next_direction)
        elif boundType=="outbound":
            for current_direction,road in signal.outRoadDict.items():
                if road is None:
                    continue
                targetVehicleList=self._createDistanceTargetOutboundVehicleList(
                    minDistanceFromInter,
                    maxDistanceFromInter,
                    road.roadId,
                )
                yield targetVehicleList,current_direction
        else:
            raise Exception("not expected")

                
    def iterateTimeTargetVehicleList(self,interId,limit_time):
        #下記のtupleをイテレータします。
        # tupleの要素0：下記の進行方向を満たし、かつ指定条件（※１）を満たすvehicleのList
        # tupleの要素1：交差点での進行方向を表すtuple（current_direction,next_direction）、current_directionから侵入しnext_directionに抜けることを表す
        #
        #指定条件（※１）
        #・limit_timeで示す制限時間内にinterIdに到達すること
        #  interIdに直接接続しているroad上のvehicleのみが対象となります。
        #・limit_timeがNoneである場合には、指定条件を無視します（maxDepthも無視）
        signal = self.signalDict[interId]
        for road in self.roadDS.interIdToInboudRoadListDict[interId]:                
            current_direction,_=signal.solveDirection(road.roadId)
            for next_direction,nextRoad in signal.outRoadDict.items():
                if nextRoad is None or next_direction==current_direction:
                    continue
                targetVehicleList=self.vehicleDS.getSortedVehicleList(road.roadId,nextRoad.roadId)
                yield targetVehicleList,(current_direction,next_direction)
    
    def createPhaseToRunDistanceDict(self,
            interId,
            limit_time,
            debug=False,
            prohibitDecreasingGoalDistance=False,
            prohibitDecreasingSpeed=True
        ):
        
        runDistanceDict={
            "pass":{},
            "stop":{},
            "stop_then_pass":{},
        }
        
        for targetVehicleList,directions in self.iterateTimeTargetVehicleList(interId,limit_time):
            for focusedSignalAssumedState in runDistanceDict:
                runDistanceDict[focusedSignalAssumedState][directions]=self._calcRunningRouteDistanceForVehicelList(
                    targetVehicleList,
                    limit_time,
                    interId,
                    focusedSignalAssumedState,
                    prohibitDecreasingGoalDistance=prohibitDecreasingGoalDistance,
                    prohibitDecreasingSpeed=prohibitDecreasingSpeed
                )
        now_phase=self.worldSignalState.signalStateDict[interId].phase
        
        phaseToRunDistanceDict={}
        for phaseId in range(1,9):
            runDistanceForStopDirection=self._sumDictValues(
                SignalState.phaseToStopDirectionListDict[phaseId],
                runDistanceDict["stop"],
            )
            runDistanceForPassDirection=self._sumDictValues(
                SignalState.phaseToPassableDirectionListDict[phaseId],
                runDistanceDict["pass" if phaseId==now_phase else "stop_then_pass"],
            )
            phaseToRunDistanceDict[phaseId]=runDistanceForStopDirection+runDistanceForPassDirection
        return phaseToRunDistanceDict
    
    @staticmethod
    def _getLeadingVehicleList(sorted_vehicles,roadLength,speedThres,distanceDiffThres):
        #sorted_vehicles(distanceが大きい順)から条件を満たす先頭集団のvechicleのリストを返します
        frontDistance=roadLength
        vehicleList=[]
        for vehicle in sorted_vehicles:         
            if vehicle.speed<speedThres:
                if (frontDistance-vehicle.distance<distanceDiffThres):
                    vehicleList.append(vehicle)
                    frontDistance=vehicle.distance
                    continue
            break
        return vehicleList

    @classmethod
    def _getStoppingVehicleList(cls,sorted_vehicles,roadLength):
        return cls._getLeadingVehicleList(sorted_vehicles,roadLength,0.0001,VEHICLE_MERGIN_METER+0.01)
    
    @classmethod
    def _getSlowVehicleList(cls,sorted_vehicles,roadLength):
        vehicleList=cls._getLeadingVehicleList(sorted_vehicles,roadLength,10,15)
        
        if len(vehicleList)>0:
            for i in range(len(vehicleList)):
                if vehicleList[-1-i].speed>0.0001:
                    break
            slowVehicleList = vehicleList[:len(vehicleList)-i]
            stoppingVehicleList = vehicleList[len(vehicleList)-i:]
            return slowVehicleList,stoppingVehicleList
        else:
            return [],[]
    
    @classmethod
    def _calcRunningRoadDistanceAndTime(cls,
            speedLimit,
            goalDistance,
            currentDistance,
            remained_time,
            speed=None
        ):
        if speed is None:
            run_time = min(remained_time,max(0,goalDistance-currentDistance)/speedLimit)
            return run_time*speedLimit,run_time
        else:
            remained_distance=max(0,goalDistance-currentDistance)
            if remained_distance==0:
                return 0,0
            else:
                run_time = min(remained_time,cls._calcRunningTime(speedLimit,remained_distance,speed))
                return min(remained_distance,cls._calcRunningDistance(speedLimit,run_time,speed)),run_time
        
    @staticmethod
    def _calcRunningDistance(vMax,t,v=None,a=5):
        if v is None:
            return vMax*t
        else:
            v0=min(vMax,v+a)
            _t=min(t,max(0,math.ceil((vMax-v0)/a)))
            return (t-_t)*vMax+_t*(v0+a*(_t-1)/2)
    @staticmethod
    def _calcRunningTime(vMax,x,v=None,a=5):
        if v is None:
            return x/vMax
        else:
            t=0
            while True:
                if x<=0:
                    break
                t+=1
                v=min(vMax,v+a)
                x-=v
            return t
        
    def _calcRunningRouteDistanceForVehicelList(self,
            vehicleList,
            limit_time,
            focusedSignalizedInterId,
            focusedSignalAssumedState,
            prohibitDecreasingGoalDistance=False,
            prohibitDecreasingSpeed=True
        ):
        
        sum_run_distance=0
        for vehicle in vehicleList:
            run_distance,inter_archived=self._calcRunningRouteDistance(
                vehicle,
                limit_time,
                focusedSignalizedInterId,
                focusedSignalAssumedState,
                prohibitDecreasingGoalDistance=prohibitDecreasingGoalDistance,
                prohibitDecreasingSpeed=prohibitDecreasingSpeed
            )
            
            if inter_archived:
                sum_run_distance+=run_distance
        return sum_run_distance
        
    def _calcRunningRouteDistance(self,
            vehicle,
            limit_time,
            focusedSignalizedInterId,
            focusedSignalAssumedState,
            prohibitDecreasingGoalDistance=False,
            prohibitDecreasingSpeed=True
        ):
        run_distance=0
        remained_time=limit_time
        interPassed=False
        interAchived=False
        route=[vehicle.roadId,vehicle.nextRoadId]
        for idx,roadId in enumerate(route):
            prevRoadId = route[idx-1] if 0<idx else None
            nextRoadId = route[idx+1] if len(route)>idx+1 else None
            road = self.roadDS.roadDict[roadId]
            roadLength=road.roadSegment.length
            startInterId=road.getStartInterId()
            endInterId=road.getEndInterId()
            
            if nextRoadId is None:
                passbale=True #TBD: taking account of possibility that all 3 lanes are occupied then not paasable 
            else:
                if endInterId==focusedSignalizedInterId:
                    passable=(focusedSignalAssumedState!="stop")
                elif interPassed:
                    passable=self.worldSignalState.isPassableRoads(endInterId,roadId,nextRoadId,self.signalDict)
                else:
                    passable=True
                    
            sorted_vehicles = self.vehicleDS.getSortedVehicleList(roadId,nextRoadId)

            #remained_time以内の時間でのroadにおける走行距離と走行時間を計算
            goalDistance = roadLength
            speedLimit=road.roadSegment.speedLimit
            if passable:
                if not prohibitDecreasingSpeed and idx==0:
                    slowVehicleList,stoppingVehicleList = self._getSlowVehicleList(sorted_vehicles,roadLength)
                    if vehicle in slowVehicleList and slowVehicleList[0]!=vehicle:
                        speedList=[v.speed for v in slowVehicleList]
                        speedLimit=sum(speedList)/len(speedList)
                    elif vehicle in stoppingVehicleList:
                        if len(slowVehicleList)!=0 or stoppingVehicleList[0]!=vehicle:
                            speedLimit=interpolate(
                                SPEED_LIMIT_IN_SLOW_VEHICLES/2,
                                SPEED_LIMIT_IN_SLOW_VEHICLES,
                                stoppingVehicleList.index(vehicle),
                                len(slowVehicleList)
                            )
            else:
                if (not prohibitDecreasingGoalDistance):
                    #信号待機列による詰まりの分、走行距離を減算する
                    if endInterId!=focusedSignalizedInterId:
                        waitingVehicleList = self._getStoppingVehicleList(sorted_vehicles,roadLength)
                        if len(waitingVehicleList)>0:
                            goalDistance = max(0,waitingVehicleList[-1].distance - VEHICLE_MERGIN_METER)
            
            runTuple=self._calcRunningRoadDistanceAndTime(
                speedLimit,
                goalDistance,
                vehicle.distance if idx==0 else 0,
                remained_time,
                vehicle.speed if idx==0 else None
            )
            #このroadでの走行距離をrun_distanceに加算し、走行時間を残り時間から減算
            run_distance+=runTuple[0]
            remained_time-=runTuple[1]
            
            #残り時間がなくなったら、走行時間の減算を停止
            if remained_time<=0:
                break
            
            if endInterId==focusedSignalizedInterId:
                interAchived=True
                if focusedSignalAssumedState=="pass":
                    interPassed=True
                elif focusedSignalAssumedState=="stop":
                    break
                elif focusedSignalAssumedState=="stop_then_pass":
                    remained_time-=max(0,PHASE_CHANGE_STOP_SEC-(limit_time-remained_time))
                    if remained_time<=0:
                        break
                    else:
                        interPassed=True
                else:
                    raise Exception("not expected")
            elif endInterId in self.worldSignalState.signalStateDict: #signalized intersection
                signalState=self.worldSignalState.signalStateDict[endInterId]
                if  signalState.tempPhase!=signalState.phase: #5秒間の全方向進行停止状態
                    remained_time-=max(0,PHASE_CHANGE_STOP_SEC-(limit_time-remained_time))
                #残り時間がなくなったら、走行時間の減算を停止
                if remained_time<=0:
                    break
                
            #f通過不可能なintersectionな場合、run_distanceの加算を停止
            if not passable:
                break
                
        return run_distance,interAchived