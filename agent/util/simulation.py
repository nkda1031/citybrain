############################
# Simulatorクラス
#  citybrainのシミュレーション環境をラップし、
#  1回のエピソードを実行するメソッドを提供します。
#    - runOneEpisodeでrunTypeを指定してエピソードを実行できます
#    - evaluateで公式のevaluate.pyと同様の方法でエピソードの実行と結果の出力を行います
#
# EpisodeRunnerクラス
#  Simulatorクラスをラップし、終了条件を指定して繰り返しエピソードを実行する機能を提供します。
# 
# ObservationsReaderクラス
#  citybrainのシミュレーション環境がstepごとに返す情報（observations,info,rewards）を処理し、
#  扱いやすい情報に変換し保持します。
#
# BridgeAgentクラス
#  citybrainで公式で指定されるagentのIFを満たし、
#  かつSimulatorクラスで定義しているrunTypeを指定したエピソード実行をサポートしたAgentです。
#  本クラスはIFを整合させるための機能のみのクラスであり、実際の処理は下記メソッドを持つactionSolverにて実装。
#    startEpisode(agent_list)
#    setRoadNet(roadNet)
#    act(observations, debug)
#    actWithRunType(observations, runType, exitType, debug)
#  actWithRunTypeは、SimulatorクラスのrunOneEpisodeをrunTypeを指定して呼び出したときに呼ばれます。
#  actは、下記の場合に呼ばれます。
#      -SimulatorクラスのrunOneEpisodeをrunTypeを指定せずに呼び出したとき
#      -Simulatorクラスのevaluateを呼びだした時
#      -公式の評価プログラムが直接actを呼び出した時
############################

from pathlib import Path
import logging
import time
import json
import os
import pprint
import shutil
import re

from evaluate import format_exception,process_roadnet,read_config

from roadnet_util import RoadNet
from time_series_util import PhaseDataSetTimeSeries,VehicleDataSetTimeSeries,VehicleDataSet
from signal_state_util import WorldSignalState,SignalState
from road_tracer import RoadTracer

logger = logging.getLogger("simulation_util.py")

def resetDir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

def exportJsonForVisualization(
        simulator_cfg_file,
        every_n=1,
        tqdm=None
    ):
    configs=Simulator.readConfig(simulator_cfg_file)
    log_dir=configs['report_log_addr']
    
    print("------ start to export simulation output as json for visualization")
    vehicleDataSetTimeSeries=VehicleDataSetTimeSeries.fromEnvLogDir(
        log_dir,
        tqdm=tqdm
    )
    vehicleDataSetTimeSeries.exportAsJson("{}/vehicle.json".format(log_dir))
    
    phaseDataSetTimeSeries=PhaseDataSetTimeSeries.fromEnvLogDir(
        log_dir,
        configs['report_log_rate'],
        every_n=every_n,
        tqdm=tqdm
    )
    phaseDataSetTimeSeries.exportAsJson("{}/phase.json".format(log_dir))
    print("------ all done")    
    
def showDelayIndex(simulator_cfg_file,tqdm=None):
    #using route
    configs=Simulator.readConfig(simulator_cfg_file)
    roadNet=RoadNet.createFromRoadNetFile(configs['road_file_addr'])
    vehicleDataSetTimeSeries = VehicleDataSetTimeSeries.fromEnvLogDir(
        configs['report_log_addr'],
        configs['max_time_epoch'],
        tqdm=tqdm
    )
    timeToDelayIndexDict = vehicleDataSetTimeSeries.calcDelayIndex(roadNet.roadDataSet)
    timeToNumServedVehiclesDict = vehicleDataSetTimeSeries.getTimeToNumServedVehiclesDict()
    pprint.pprint(timeToNumServedVehiclesDict)
    pprint.pprint(timeToDelayIndexDict)
    
class EpisodeRunner:
    def __init__(self,
            actionSolver,
            CBEngine_rllib_class,
            simulator_cfg_file,
            metric_period,
            debug=False
        ):
        self.actionSolver=actionSolver
        self.agent=BridgeAgent(actionSolver,debug=debug)
        self.sim=Simulator(
            CBEngine_rllib_class,
            simulator_cfg_file,
            metric_period,
        )
        self.log_dir=self.sim.simulator_configs['report_log_addr']
        self.evalRecordList=[]
    
    def export(self,tqdm=None):
        print("------ start to export simulation output as json for visualization")
        vehicleDataSetTimeSeries=VehicleDataSetTimeSeries.fromEnvLogDir(self.log_dir,tqdm=tqdm)
        vehicleDataSetTimeSeries.exportAsJson("{}/vehicle.json".format(self.log_dir))
        
        phaseDataSetTimeSeries=PhaseDataSetTimeSeries.fromEnvLogDir(self.log_dir,int(self.sim.simulator_configs['report_log_rate']),tqdm=tqdm)
        phaseDataSetTimeSeries.exportAsJson("{}/phase.json".format(self.log_dir))  
    
    def _runEpisode(self,
            runType,
            pbar,
            numRunnedEpisodes,
            earlyStoppingDelayIndex=None
        ):
        if pbar is not None:
            pbar.reset()
        timeToNumServedVehiclesDict,  timeToDelayIndexDict = self.sim.runOneEpisode(
            self.agent,
            runType,
            earlyStoppingDelayIndex=earlyStoppingDelayIndex,
            pbar=pbar
        )
        if len(timeToDelayIndexDict)>0:
            lastTime=max(timeToDelayIndexDict.keys())
            trainStep,trainLoss = self.actionSolver.getTrainProgress()
            if runType=="eval":
                self.evalRecordList.append((
                    lastTime,
                    timeToNumServedVehiclesDict[lastTime],
                    timeToDelayIndexDict[lastTime],
                    trainStep,
                    trainLoss
                ))
            print("{} {}: t={}, served={}, DI={:.3}, buf={}, train {}, loss={:.3}".format(
                runType,
                numRunnedEpisodes+1,
                lastTime,
                timeToNumServedVehiclesDict[lastTime],
                timeToDelayIndexDict[lastTime],
                self.actionSolver.getBufferLength(),
                trainStep,
                trainLoss,
            ))
        else:
            raise Exception("evaluation does not exist. please set metric_period smaller")
            
    def runLoop(self,
            runType,
            earlyStoppingDelayIndex=None,
            *,
            eval_every_n_episode=None,
            breakReplayBufferLength=None,
            breakNumEpsiode=None,
            tqdm=None
        ):
        if breakNumEpsiode is None and runType=="eval":
            breakNumEpsiode=1
            
        def _loop(pbar=None):
            episodes=0
            evalEpisodes=0
            start_time = time.time()
            while True:
                self._runEpisode(runType,pbar,episodes,earlyStoppingDelayIndex=earlyStoppingDelayIndex)
                episodes+=1
                if eval_every_n_episode is not None and runType=="train" and episodes % eval_every_n_episode==0:
                    self._runEpisode("eval",pbar,evalEpisodes,earlyStoppingDelayIndex=earlyStoppingDelayIndex)
                    evalEpisodes+=1
                if breakReplayBufferLength is not None:
                    bufLen=self.actionSolver.getBufferLength()
                    if bufLen>=breakReplayBufferLength:
                        break
                if breakNumEpsiode is not None and breakNumEpsiode==episodes:
                    break    
            end_time = time.time()
            print("--- episode loop completed------------------")
            if runType =="train":
                print("train {} episodes, eval {} episodes, total {:.1} sec".format(episodes,evalEpisodes,end_time-start_time))
            else:
                print("{} {} episodes, total {:.1} sec".format(runType,episodes,end_time-start_time))
            print("---------------------")
        
        if tqdm is not None:
            with tqdm() as pbar:
                _loop(pbar)
        else:
            _loop()
            
class BridgeAgent():
    #citybrainチャレンジで要求されるインタフェース（act,load_agent_list,load_roadnet）及び
    #Simulatorクラスで追加で要求するインタフェース（actWithRunType）の呼び出しを受ける。
    #実際の処理はactionSolverにて実装。
    #actionSolverは下記のメソッドを実装すること
    #  startEpisode(agent_list)
    #  setRoadNet(roadNet)
    #  act(observations, debug)
    #  actWithRunType(observations, runType, exitType, debug)
    def __init__(self,actionSolver=None,debug=False):
        self.debug=debug
        self.printActionStat=False
        
        #initialize
        self.agent_list = None
        
        if actionSolver is None:
            raise Exception("TBD: load from checkpoint")
        else:
            self.actionSolver=actionSolver
        self.roadNet=None
    ################################
    def load_agent_list(self,agent_list):
        self.agent_list = list(map(int, agent_list))
        self.actionSolver.startEpisode(self.agent_list)
    def load_roadnet(self,intersections, roads, agents):
        self.roadNet=RoadNet.createFromProcessRoadNet(intersections, roads, agents)
        self.actionSolver.setRoadNet(self.roadNet)
    ################################
    def act(self, obs):
        obs["observations"]={
            int(k):v
                for k,v in obs["observations"].items()
        }
        observations=ObservationsReader(
            obs['observations'],
            obs['info'],
            self.roadNet
        )
        actions=self.actionSolver.act(observations, debug=self.debug)
        if self.printActionStat:
            print(self.actionSolver.getStatMessage())
        return actions
    def actWithRunType(self, obs, runType, exitType):
        obs["observations"]={
            int(k):v
                for k,v in obs["observations"].items()
        }
        observations=ObservationsReader(
            obs['observations'],
            obs['info'],
            self.roadNet
        )
        actions=self.actionSolver.actWithRunType(observations,runType,exitType=exitType, debug=self.debug)
        if self.printActionStat:
            print(self.actionSolver.getStatMessage())
        return actions
    
class _BaseActionSolver:
    def __init__(self):
        self.roadNet=None
        self.worldSignalState=None
        self.lastEpisodeWorldSignalState=None
        self.firstActInEpisodeCalled=False
        
        self.prevObservations=None
        self.prevWorldSignalState=None
        self.prev2Observations=None
        self.prev2WorldSignalState=None
        
        self.firstVehicleDict={}#時間的に最初に出現したvehicleを保存
        
    def setRoadNet(self,roadNet):
        self.roadNet=roadNet
            
    def startFirstEpisode(self,signalizedInterIdList):
        pass #please override to do something
    def startFollowingEpisode(self,signalizedInterIdList):
        pass #please override to do something
        
    def startEpisode(self,signalizedInterIdList):
        
        #episodeの度に作り直す
        self.firstVehicleDict={}
        worldSignalState=WorldSignalState()
        for interId in signalizedInterIdList:
            worldSignalState.add(SignalState(interId))
        
        self.prevActCountInEpisode=0
        if self.worldSignalState is None: #first episode
            self.worldSignalState=worldSignalState
            self.startFirstEpisode(signalizedInterIdList)
        else:
            self.lastEpisodeWorldSignalState=self.worldSignalState
            self.worldSignalState=worldSignalState
            self.startFollowingEpisode(signalizedInterIdList)
            
    def getTrainProgress(self):
        return 0,0.
    
    def getBufferLength(self):
        return 0
    
    def getStatMessage(self):
        return "(no stat)"
    
    def getExtraMessage(self):
        return None
    
    def act(self,
            observations,
            debug=False
        ):
        return self.actWithRunType(observations,"eval",None,debug)
    
    def actWithRunType(self,
            observations,
            runType="eval",
            exitType=False,
            debug=False
        ):
        for vehicleId,vehicle in observations.vehicleDS.existingVehicleDict.items():
            if vehicleId not in self.firstVehicleDict:
                self.firstVehicleDict[vehicleId]=vehicle
            vehicle.setOriginalVehicle(self.firstVehicleDict[vehicleId])
                
        clonedWorldSignalState=self.worldSignalState.clone()
        ############################
        #この中でworldSignalState中のsignalStateは変化する
        actions=self.decideActions(
            observations,
            self.prevActCountInEpisode,
            runType,
            exitType,
            debug=debug)
        ############################
        self.prevActCountInEpisode+=1
        
        self.prev2Observations=self.prevObservations
        self.prev2WorldSignalState=self.prevWorldSignalState
        
        self.prevObservations=observations
        self.prevWorldSignalState=clonedWorldSignalState
        
        return actions
        
        raise Exception("_BaseActionSolver is abstarct class. please create subclass and override actWithRunType")
        
    def _createRoadTracer(self,observations):
        return self._createRoadTracerSub(self.worldSignalState,observations)
    def _createPrevRoadTracer(self):
        return self._createRoadTracerSub(self.prevWorldSignalState,self.prevObservations)
    def _createPrev2RoadTracer(self):
        return self._createRoadTracerSub(self.prev2WorldSignalState,self.prev2Observations)
    def _createRoadTracerSub(self,worldSignalState,observations):
        return RoadTracer(
            worldSignalState,
            observations.vehicleDS,
            self.roadNet.intersectionDataSet.signalDict,
            self.roadNet.roadDataSet,
        )
    
class FixedActionSolver(_BaseActionSolver):
    def __init__(self,
             timeToActionsDict
        ):
        super().__init__()
        self.timeToActionsDict=timeToActionsDict
        self.inited=False
        
    def decideActions(self,
            observations,
            prevActCountInEpisode,
            runType="eval",
            exitType=False,
            debug=False
        ):
        actions=self.timeToActionsDict[observations.current_time] if observations.current_time in self.timeToActionsDict else {}
        return actions
    
class LogActionSolver(_BaseActionSolver):
    def __init__(self,
            action_log_path,
        ):
        super().__init__()
        with open(action_log_path) as f:
            actionLogDict=json.load(f)
            
        self.actionLogDict={}
        for t,actions in actionLogDict.items():
            if int(t) not in self.actionLogDict:
                self.actionLogDict[int(t)]={}
            for interId,phase in actions.items():
                self.actionLogDict[int(t)][int(interId)]=phase
        
    def decideActions(self,
            observations,
            prevActCountInEpisode,
            runType="eval",
            exitType=False,
            debug=False
        ):
        actions=self.actionLogDict[observations.current_time] if observations.current_time in self.actionLogDict else {}
        return actions
    
class ObservationsReader():
    def __init__(self,observations,info,roadNet):
        self.interIdToObservationDict = observations
        self.current_time=info["step"]
        self.vehicleDS = VehicleDataSet.fromObservationInfo(info,self.current_time,roadNet)
        
def _create_result(total_served_vehicles=None,delay_index=None,error_msg=""):
    # result to be written in out/result.json
    if total_served_vehicles is None or delay_index is None:
        result = {
            "success": False,
            "error_msg": error_msg,
            "data": {
                "total_served_vehicles": -1,
                "delay_index": -1
            }
        }
    else:
        result = {
            "success": True,
            "error_msg": error_msg,
            "data": {
                "total_served_vehicles": total_served_vehicles,
                "delay_index": delay_index
            }
        }
    return result

class Simulator:
    def __init__(self,
            CBEngine_rllib_class,
            simulator_cfg_file,
            metric_period,
            gym_configs=None,
            thread_num=1,
            vehicle_info_path="log",
            output_dir="out"
        ):
        if gym_configs is None:
            gym_configs = {
                'observation_features':['lane_vehicle_num','classic'],
                'observation_dimension':40,
                'custom_observation' : False
            }
            
        # get gym instance
        self.simulator_configs = self.readConfig(simulator_cfg_file)
        
        flow_index = int(re.findall('[0-9]+',self.simulator_configs['vehicle_file_addr'])[1])
        log_path = Path(vehicle_info_path) / str(flow_index)
        scores_dir = Path(output_dir) / str(flow_index)
        
        self.env = CBEngine_rllib_class({
            "simulator_cfg_file": simulator_cfg_file,
            "thread_num": thread_num,
            "gym_dict": gym_configs,
            "metric_period": metric_period,
            "vehicle_info_path":log_path
        })
        
        roadnet_path=self.simulator_configs['road_file_addr']
        self.intersections, self.roads, self.agents = process_roadnet(roadnet_path)
        
        self.roadNet=RoadNet.createFromRoadNetFile(roadnet_path)
        
        self.env.set_log(1)
        self.env.set_info(1)
        
    @staticmethod
    def readConfig(simulator_cfg_file):
        config_raw=read_config(simulator_cfg_file)
        return {
            'max_time_epoch':int(config_raw['max_time_epoch']),
            'report_log_addr':config_raw['report_log_addr'],
            'report_log_mode':config_raw['report_log_mode'],
            'report_log_rate':int(config_raw['report_log_rate']),
            'road_file_addr':config_raw['road_file_addr'],
            'start_time_epoch':int(config_raw['start_time_epoch']),
            'vehicle_file_addr':config_raw['vehicle_file_addr'],
            'warning_stop_time_log':int(config_raw['warning_stop_time_log']),
        }
    
    @staticmethod
    def _act(agent,runType,obs,exitType=None):
        if runType is None:
            return agent.act(obs)
        else:
            return agent.actWithRunType(obs,runType,exitType)
    
    def _run(self,agent,runType=None,earlyStoppingDelayIndex=None,pbar=None):
        observations = self.env.reset()
        info = {'step':0}
        
        agent_id_list=list(observations.keys())
        agent.load_agent_list(agent_id_list)
        agent.load_roadnet(self.intersections, self.roads, self.agents)
        
        def _roop(observations,info):
            rewards = None
            step=0
            while True:
                actions=self._act(agent,runType,{
                    'observations':observations,
                    'info':info,
                })
                observations, rewards, dones, info = self.env.step(actions)
                step+=1
                info['step'] = step
                
                agent_id_list=list(observations.keys())
                for agent_id in agent_id_list:
                    if(dones[agent_id]):
                        return observations, rewards, dones, info,True
                
                if pbar is not None:
                    pbar.update(1)
                if earlyStoppingDelayIndex is not None:
                    currentTime=step*10
                    latestInfoStepTime=currentTime-1
                    log_path="{}/info_step {}.log".format(self.simulator_configs['report_log_addr'],latestInfoStepTime)
                    if os.path.exists(log_path):
                        if pbar is not None:
                            pbar.set_description("start to calc DI")
                        delayIndex = VehicleDataSet.fromEnvLogFile(log_path,latestInfoStepTime).calcDelayIndex(self.roadNet.roadDataSet)
                        if pbar is not None:
                            pbar.set_description("DI={:.3} at {}".format(float(delayIndex),latestInfoStepTime))
                        if delayIndex>earlyStoppingDelayIndex:
                            logger.warning("early stop at time {}: delay index {:.3} is higher than {}".format(
                                latestInfoStepTime,
                                delayIndex,
                                earlyStoppingDelayIndex
                            ))
                            return observations, rewards, dones, info,False
                        
        observations, rewards, dones, info, completed=_roop(observations,info)
        self._act(agent,runType,{
            'observations':observations,
            'info':info,
        },exitType="done" if completed else "stop")
    
    def runOneEpisode(self,
            agent,
            runType=None,
            earlyStoppingDelayIndex=None,
            pbar=None
        ):
        resetDir(self.simulator_configs['report_log_addr'])
        logger.info("*" * 40)
        logger.info("start to simulate")
        
        self._run(agent,runType=runType,earlyStoppingDelayIndex=earlyStoppingDelayIndex,pbar=pbar)
        vehicleDataSetTimeSeries = VehicleDataSetTimeSeries.fromEnvLogDir(
            self.simulator_configs['report_log_addr'],
            self.simulator_configs['max_time_epoch']
        )
        
        timeToDelayIndexDict = vehicleDataSetTimeSeries.calcDelayIndex(self.roadNet.roadDataSet)
        timeToNumServedVehiclesDict = vehicleDataSetTimeSeries.getTimeToNumServedVehiclesDict()
        return timeToNumServedVehiclesDict,  timeToDelayIndexDict
    
    def evaluate(self,
            agent,
            scores_out_dir="out",
            printDelayIndexTimeSeries=False,
            earlyStoppingDelayIndex=None,
            tqdm=None
        ):
        def write_result(result,out_path):
            json.dump(result, open(out_path, 'w'), indent=2)
            
        out_path = Path(scores_out_dir) / "scores.json"
        # simulation
        start_time = time.time()
        try:
            if tqdm is not None:
                with tqdm() as pbar:
                    timeToNumServedVehiclesDict,timeToDelayIndexDict = self.runOneEpisode(
                        agent,
                        earlyStoppingDelayIndex=earlyStoppingDelayIndex,
                        pbar=pbar
                    )
            else:
                timeToNumServedVehiclesDict,timeToDelayIndexDict = self.runOneEpisode(
                    agent,
                    earlyStoppingDelayIndex=earlyStoppingDelayIndex
                )
            lastTime=max(timeToDelayIndexDict.keys())
            write_result(
                _create_result(timeToNumServedVehiclesDict[lastTime],timeToDelayIndexDict[lastTime]),
                out_path
            )
        except Exception as e:
            write_result(
                _create_result(error_msg=format_exception(e)),
                out_path
            )
            raise Exception("error when running simulation")
            
        end_time = time.time()
        logger.info(f"total evaluation cost {end_time-start_time} s")
        
        print("-----result----")
        print("last time :",lastTime)
        if printDelayIndexTimeSeries:
            pprint.pprint(timeToNumServedVehiclesDict)
            pprint.pprint(timeToDelayIndexDict)
        else:
            print("num served vehicles :",timeToNumServedVehiclesDict[lastTime])
            print("delay index :",timeToDelayIndexDict[lastTime])
        print("---------------", flush=True)
        
        logger.info("*" * 40)
        logger.info("Evaluation complete")
        return timeToNumServedVehiclesDict[lastTime],timeToDelayIndexDict[lastTime],lastTime