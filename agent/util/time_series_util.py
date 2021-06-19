from pathlib import Path
import json
import os
import re

import numpy as np

class Vehicle:
    def __init__(self,vehicle_raw,currentTime,vehicleId,roadNet=None):
        #### vehicle data definition
        # Not Changable (time step independent)  
        # "vehicleId": int
        #     vehicle id
        # "startTime": float
        #     Time of creation of this vehicle.
        # "minTravelTime": (optional) float
        #     Travel time of this vehicle assuming no traffic signal and other vehicle exists.
        #
        # Changable (time step dependent)  
        # "distance": float
        #     The distance from this vehicle to the start point of current road.
        # "lane": int
        #     Current lane of this vehicle. usually 0,1 or 2 .
        # "roadId": int
        #     Current road of this vehicle.
        # "nextRoadId":  (optional) int
        #     Next road of this vehicle, that is expected from lane (that may be not correct because of changing lane behaivior)
        # "speed": float
        #     Current speed of this vehicle.
        # "currentTime": int
        #     Time of observation of this vehicle.
        # "route": (optional) List of int
        #     Route of this vehicle (starting from current to destination).        
        self.vehicleId=vehicleId
        self.startTime=vehicle_raw["start_time"]
        
        self.distance=vehicle_raw["distance"]
        self.lane=vehicle_raw["drivable"]%100
        self.roadId=vehicle_raw["road"]
        if roadNet is not None:
            self.nextRoadId=roadNet.getNextRoadId(self.roadId,self.lane)
        else:
            self.nextRoadId=None
        self.speed=vehicle_raw["speed"]
        self.goalTime=vehicle_raw["step"] if vehicle_raw["step"] is not None and vehicle_raw["step"]<currentTime else None
        self.currentTime=currentTime
        
        self.route=vehicle_raw["route"] if "route" in vehicle_raw else None
        self.minTravelTime=vehicle_raw["t_ff"] if "t_ff" in vehicle_raw else None
            
    def exportAsDict(self):
        return {
            'vehicleId':self.vehicleId,
            'startTime':self.startTime,
            'distance':self.distance,
            'lane':self.lane,
            'roadId':self.roadId,
            'speed':self.speed,
            'currentTime':self.currentTime,
            'goalTime':self.goalTime,
        }
    def calcRunDistance(self,prevVehicel,roadDataSet,normalize=False):
        return prevVehicel.calcRemaindLengthToGoal(roadDataSet,normalize=normalize) - self.calcRemaindLengthToGoal(roadDataSet,normalize=normalize)
    def calcRemaindLengthToGoal(self,roadDataSet,normalize=False):
        return roadDataSet.calcRemaindLengthToGoal(self,normalize=normalize)
    def calcOriginalLengthToGoal(self,roadDataSet):
        return roadDataSet.calcRemaindLengthToGoal(self.originalVehicle,normalize=False)
    def setOriginalVehicle(self,vehicle):
        self.originalVehicle=vehicle
    def __str__(self):
        return "id={},x={},v={},goal={}".format(
            self.vehicleId,
            self.distance,
            self.speed,
            self.goalTime
        )

class VehicleDataSet():
    def __init__(self,allVehicleDict):
        # allVehicleDict retains vehicles informations including ones that have arrived at its destination.
        self.allVehicleDict = allVehicleDict
        self.existingVehicleDict = {
            vehicleId:vehicle
                for vehicleId,vehicle in allVehicleDict.items()
                    if vehicle.goalTime is None
        }
        
        vehicles_on_road={}
        for vehicle in self.existingVehicleDict.values():
            road_id = vehicle.roadId
            if road_id not in vehicles_on_road:
                vehicles_on_road[road_id]={}
            next_road_id=vehicle.nextRoadId
            if next_road_id not in vehicles_on_road[road_id]:
                vehicles_on_road[road_id][next_road_id]=[]
            vehicles_on_road[road_id][next_road_id].append(vehicle)
            
        for vehicles_dests in vehicles_on_road.values():
            for vehicles_dest in vehicles_dests.values():
                vehicles_dest.sort(key = lambda vehicle:vehicle.distance,reverse=True)
        self.vehiclesOnRoadDict=vehicles_on_road

    def getSortedVehicleList(self,route):
        #Parameter:
        #  route: List of int
        #    it specifies roadIDs of future route of a vehicle
        #    [current_roadId,next_roadId,next_next_roadId,...]
        #
        #Return: List of Vehicle
        #  this func returns Vehicles that match 'route'. list is sorted by distance of vehicle
        currentRoadId=route[0]
        if len(route)==1:
            if currentRoadId in self.vehiclesOnRoadDict:
                vehicleList=[]
                for sortedVehicleList in self.vehiclesOnRoadDict[currentRoadId].values():
                    vehicleList+=sortedVehicleList
                vehicleList.sort(key = lambda vehicle:vehicle.distance,reverse=True)
                return vehicleList
        else:
            nextRoadId=route[1]
            if currentRoadId in self.vehiclesOnRoadDict and nextRoadId in self.vehiclesOnRoadDict[currentRoadId]:
                sortedVehicleList=self.vehiclesOnRoadDict[currentRoadId][nextRoadId]

                len_route=len(route)
                if len_route>=3:
                    #clip according to route
                    return [
                        vehicle
                            for vehicle in sortedVehicleList
                                if vehicle.route[:len_route]==route
                    ]
                else:
                    return sortedVehicleList
        return []

    def calcDelayIndex(self,roadDataSet):
        #正しくDelay Indexを計算するには、既にゴールしたvehicleも計算対象にする必要がある。
        #fromEnvLogFileから生成した場合は、既にゴールしたvehicleもallVehicleDictに含まれるが、
        #fromObservationInfoから生成した場合は、既にゴールしたvehicleが含まれない。
        
        if len(self.allVehicleDict)==0:
            return 1
        
        delayIndexList = []
        for vehicle in self.allVehicleDict.values():
            delayIndexList.append(roadDataSet.calcDelayIndex(vehicle))
        delay_index = np.mean(delayIndexList)
        return delay_index
    
    @staticmethod
    def _simplifyVehicle(vehicle_raw):
        return {
            "distance":vehicle_raw["distance"][0],
            "drivable":int(vehicle_raw["drivable"][0]),
            "road":int(vehicle_raw["road"][0]),
            "speed":vehicle_raw["speed"][0],
            "start_time":vehicle_raw["start_time"][0],
            "step":vehicle_raw["step"][0] if "step" in vehicle_raw else None,
        }
    
    @classmethod
    def fromObservationInfo(cls,info,currentTime,roadNet):
        vehicleDict = {
            vehicleId : Vehicle(
                    cls._simplifyVehicle(vehicle_raw),
                    currentTime,
                    vehicleId,
                    roadNet
                )
                for vehicleId,vehicle_raw in info.items()
                    if vehicleId!="step"
        }
        return VehicleDataSet(vehicleDict)
        
    @staticmethod
    def fromEnvLogFile(file_path,current_time):
        #### log format: example
        # 32182
        # for vehicle 0
        # distance : 586
        # drivable : 378201
        # road : 3782
        # route : 3782 3102 3026 3126 3022 4194 3990 4171 4096 2133 2136 2138 2122 3325 2890 2895 3488 3645 465 35 38 184 310
        # speed : 0
        # start_time : 0
        # t_ff : 800.691
        # step : 398
        # -----------------
        # for vehicle 1
        # distance : 28
        # drivable : 151901
        # road : 1519
        # route : 1519
        # speed : 16.6667
        # start_time : 0
        # t_ff : 335.289
        # step : 398        
        # -----------------    
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = list(map(lambda x: x.rstrip('\n').split(' '), lines))
        vehicleDict = {}
        for i in range(len(lines)):
            line = lines[i]
            if(line[0] == 'for'):
                vehicleId = int(line[2])
                vehicleDict[vehicleId] = Vehicle(
                    {
                        'distance': float(lines[i + 1][2]),
                        'drivable': int(float(lines[i + 2][2])),
                        'road': int(float(lines[i + 3][2])),
                        'route': list(map(int, list(map(float, lines[i + 4][2:])))),
                        'speed': float(lines[i + 5][2]),
                        'start_time': float(lines[i + 6][2]),
                        't_ff': float(lines[i+7][2]),
                        'step':int(lines[i+8][2])
                    },
                    current_time,
                    vehicleId,
                )
        return VehicleDataSet(vehicleDict)

class VehicleDataSetTimeSeries():
    def __init__(self,timeToVehicleDataSetDict):
        self.timeToVehicleDataSetDict=timeToVehicleDataSetDict

    def getTimeToNumServedVehiclesDict(self):   
        timeToNumServedVehiclesDict = {}
        for currentTime, vehicleDS in self.timeToVehicleDataSetDict.items():
            timeToNumServedVehiclesDict[currentTime] = len(list(vehicleDS.allVehicleDict.keys()))
        return timeToNumServedVehiclesDict
        
    def calcDelayIndex(self,roadDataSet):        
        timeToDelayIndexDict = {}
        for currentTime,vehicleDS in self.timeToVehicleDataSetDict.items():
            timeToDelayIndexDict[currentTime] = vehicleDS.calcDelayIndex(roadDataSet)
        return timeToDelayIndexDict

    def exportAsJson(self,out_path):
        def write(dic):
            print("start writing json file: "+out_path)
            json.dump(dic, open(out_path, 'w'))
        timeList=sorted(self.timeToVehicleDataSetDict.keys())
        write(
            {
                t:{
                    vehicleId:vehicle.exportAsDict()
                        for vehicleId,vehicle in self.timeToVehicleDataSetDict[t].existingVehicleDict.items()
                } for t in timeList
            }
        )
        
    @staticmethod
    def fromEnvLogDir(log_dir,max_time_epoch=None,tqdm=None):
        # read log files under specified dir
        timeToVehicleDataSetDict = {}

        for dirpath, dirnames, file_names in os.walk(log_dir):
            target_file_name_li=[f for f in file_names if f.endswith(".log") and f.startswith('info_step')]
            if tqdm is not None:
                target_file_name_li=tqdm(target_file_name_li)
            for file_name in target_file_name_li:
                pattern = '[0-9]+'
                currentTime = list(map(int, re.findall(pattern, file_name)))[0]
                if(max_time_epoch is not None and currentTime > max_time_epoch):
                    continue
                timeToVehicleDataSetDict[currentTime] = VehicleDataSet.fromEnvLogFile(Path(log_dir) / file_name,currentTime)
        return VehicleDataSetTimeSeries(timeToVehicleDataSetDict)

class PhaseDataSet():
    def __init__(self,phaseList,vehicleList):
        self.phaseList = phaseList
        self.vehicleList = vehicleList

    @staticmethod
    def fromEnvLogFile(file_path,current_time):
        with open(file_path, 'r') as f:
            loadedList = json.load(f)
        return PhaseDataSet(loadedList[1],loadedList[0])
        
class PhaseDataSetTimeSeries():
    def __init__(self,timeToPhaseDataSetDict):
        self.timeToPhaseDataSetDict=timeToPhaseDataSetDict
        
    def exportAsJson(self,out_path):
        def write(dic):
            print("start writing json file: "+out_path)
            json.dump(dic, open(out_path, 'w'))
        timeList=sorted(self.timeToPhaseDataSetDict.keys())
        write(
            {
                t:{
                    "phaseList":self.timeToPhaseDataSetDict[t].phaseList,
                    "vehicleList":self.timeToPhaseDataSetDict[t].vehicleList
                }
                    for t in timeList
            }
        )
        
    @staticmethod
    def fromEnvLogDir(log_dir,log_rate,every_n=1,max_time_epoch=None,tqdm=None):
        # read log files under specified dir
        timeToPhaseDataSetDict = {}

        for dirpath, dirnames, file_names in os.walk(log_dir):
            target_file_name_li=[f for f in file_names if f.endswith(".json") and f.startswith('time')]
            if tqdm is not None:
                target_file_name_li=tqdm(target_file_name_li)
            for file_name in target_file_name_li:
                pattern = '[0-9]+'
                idx=list(map(int, re.findall(pattern, file_name)))[0]
                if idx%every_n==0:
                    currentTime = idx*int(log_rate)
                    if log_rate>1:
                        currentTime+=1 #may be bug of gym simulator. if log_rate>1, files idx starts from 0, but it is for time step 1
                    if(max_time_epoch is not None and currentTime >= max_time_epoch):
                        continue
                    timeToPhaseDataSetDict[currentTime] = PhaseDataSet.fromEnvLogFile(Path(log_dir) / file_name,currentTime)
        return PhaseDataSetTimeSeries(timeToPhaseDataSetDict)
        