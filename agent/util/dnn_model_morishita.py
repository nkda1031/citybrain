import math
import random
import copy
import itertools

import numpy as np
import tensorflow as tf
from tensorflow import keras

from simulation import BridgeAgent,_BaseActionSolver
from signal_state_util import WorldSignalState,SignalState,LaneVehicleNumCalc
from strategy_runner import StrategyRunner
from road_tracer import RoadTracer
from dnn_input import InputArrayCreator

RUN_DISTANCE_SCALER=0.1
NORMALIZED_RUN_DISTANCE_SCALER=50

def softmax(x,axis=1):
    x_max = np.max(x,axis=axis,keepdims=True)
    e_x = np.exp(x - x_max) #subtracts each row with its max value    
    return e_x / np.sum(e_x,axis=axis,keepdims=True)

class SavedRunDistanceModel():
    def __init__(self,
            savedModelPath,
            numSegmentInbound=9,
            numSegmentOutbound=9,
            segmentLength=25,
            signatureName="embedding",
            inputName="inputs",
            outputName="output_3"
        ):
        self.inputCreator=InputArrayCreator(
            numSegmentInbound,
            numSegmentOutbound,
            segmentLength,
        )
        self.inputName=inputName
        self.outputName=outputName
        
        if tf.__version__.split(".")[0]=="1":
            self.sess=tf.compat.v1.Session()
            self.imported = tf.compat.v1.saved_model.load(self.sess,("serve",),savedModelPath)
            self.signature=self.imported.signature_def[signatureName]
            print("saved model input signature",self.signature.inputs[inputName])
            print("saved model output signature",self.signature.outputs[outputName])
        else:
            self.imported = tf.saved_model.load(savedModelPath)
            self.imported_func = self.imported.signatures[signatureName]
            print("saved model signature",self.imported_func)
    
    def calcPhaseProbAll(self,
            tracer,
        ):
        inputs,interIds = self.inputCreator.createAllSignalInputForEmbeddingLayer(tracer)
        BATCH=64
        phaseProbList=[]
        if tf.__version__.split(".")[0]=="1":
            with self.sess.as_default() as sess:
                for i in range(math.ceil(len(interIds)/BATCH)):
                    input_info=self.signature.inputs[self.inputName]
                    output_info=self.signature.outputs[self.outputName]
                    _eSumRunDist2Batch = sess.run(output_info.name, feed_dict={input_info.name: inputs[i*BATCH:(i+1)*BATCH,:]})
                    phaseProb = softmax(_eSumRunDist2Batch,axis=1) #shape = [BATCH,8,1]
                    phaseProbList.append(phaseProb[:,:,0])
            phaseProbAll=np.concatenate(phaseProbList,axis=0)# shape = [NumInterId,8]        
            return phaseProbAll,interIds
        else:
            for i in range(math.ceil(len(interIds)/BATCH)):
                _eSumRunDist2Batch = self.imported_func(inputs[i*BATCH:(i+1)*BATCH,:])[self.outputName]
                phaseProb = tf.nn.softmax(_eSumRunDist2Batch,axis=1) #shape = [BATCH,8,1]
                phaseProbList.append(phaseProb[:,:,0])

            phaseProbAll=tf.concat(phaseProbList,axis=0)# shape = [NumInterId,8]        
            return phaseProbAll.numpy(),interIds

class RunDistanceDataSetDecoder:
    def __init__(self, sectionNum):
        self.feature_description = {
            'startPhase': tf.io.FixedLenFeature([8*9], tf.int64),                     # road1~8 x 9dim one-hot
            'endPhase': tf.io.FixedLenFeature([8*9], tf.int64),                       # road1~8 x 9dim one-hot
            'startIntersectionPos': tf.io.FixedLenFeature([8*5], tf.int64),           # road1~8 x 5dim one-hot
            'endIntersectionPos': tf.io.FixedLenFeature([8*5], tf.int64),             # road1~8 x 5dim one-hot
            'speedLimit': tf.io.FixedLenFeature([8*1], tf.float32),                   # road1~8 (x float)
            'permissibleMove': tf.io.FixedLenFeature([8*3*3], tf.int64),              # lane1~24 x 3dim
            'stackingVehicles': tf.io.FixedLenFeature([8*3*1], tf.int64),             # lane1~24 x 1
            'stackingVehiclesSpeed': tf.io.FixedLenFeature([8*3*1], tf.float32),      # lane1~24 x 1
            'vehiclesInfo': tf.io.FixedLenFeature([8*3*sectionNum*4], tf.float32),    # lane1~24 x ceil(max_len/5) x 4
            'nextPhase': tf.io.FixedLenFeature([9], tf.int64),                        # 9dim one-hot
            'totalTravelDistance': tf.io.FixedLenFeature([], tf.float32)              # (float)
        }

    def load(self, tfrecordPath):
        def _parse(example_proto):
            parsed_dataset = tf.io.parse_single_example(example_proto, self.feature_description)
            
            keepingAction= 1. if parsed_dataset['action']==parsed_dataset['nextAction'] else 0.
            
            x = tf.concat([
                tf.cast(parsed_dataset['interSignalized'],tf.float32),
                tf.cast(parsed_dataset['vehicleNumInbound'],tf.float32),
                parsed_dataset['vehicleSpeedInbound'],
                parsed_dataset['normalizedRoadLengthInbound'],
                parsed_dataset['roadSpeedLimitInbound'],
                tf.cast(parsed_dataset['vehicleNumOutbound'],tf.float32),
                parsed_dataset['vehicleSpeedOutbound'],
                parsed_dataset['normalizedRoadLengthOutbound'],
                parsed_dataset['roadSpeedLimitOutbound'],
                tf.expand_dims(tf.cast(parsed_dataset['oldPhase'],tf.float32),axis=0),
                tf.expand_dims(tf.cast(parsed_dataset['action'],tf.float32),axis=0),
                tf.expand_dims(tf.cast(keepingAction,tf.float32),axis=0),
            ],axis=0)
            y = tf.concat([
                tf.cast(parsed_dataset['nextVehicleNumOutbound'],tf.float32),
                parsed_dataset['nextVehicleSpeedOutbound'],
                tf.expand_dims(parsed_dataset['vehicleSumRunDistance1']*RUN_DISTANCE_SCALER,axis=0),
                tf.expand_dims(keepingAction*parsed_dataset['vehicleSumRunDistance2']*RUN_DISTANCE_SCALER,axis=0),
            ],axis=0)
            return x,y
        
        raw_dataset = tf.data.TFRecordDataset(tfrecordPath)
        return raw_dataset.map(_parse)

def _float_feature(value):
    """float / double 型から float_list を返す"""
    if isinstance(value,list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """bool / enum / int / uint 型から Int64_list を返す"""
    if isinstance(value,list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))    
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))    
    
class DataSetCreationActionSolver(_BaseActionSolver):
    def __init__(self,
            strategy=None,
        ):
        super().__init__()
        #self.includePrevRoadForInbound=True

        if strategy is not None:
            self.strategyRunner=StrategyRunner(strategy)
        self.strategy=strategy
        self.exampleList=[]
        self.prevWorldSignalState=None
        
        self.max_laneNum = 0
        self.sectionLen = 5.0
        self.sectionNum = 0
        self.inter2roads = {}
        self.agentStaticInfo = {}
        self.road2vehicles = {}
    
    def setRoadNet(self,roadNet):
        self.roadNet=roadNet
        if self.strategy is not None:
            self.strategy.setRoadNet(roadNet)
            
        self._createStaticExample()
        
    def startFirstEpisode(self,signalizedInterIdList):
        if self.strategy is not None:
            self.strategy.setWorldSignalState(self.worldSignalState)
    def startFollowingEpisode(self,signalizedInterIdList):
        if self.strategy is not None:
            self.strategy.setWorldSignalState(self.worldSignalState)
        
    def getBufferLength(self):
        return len(self.exampleList)
    
    def _createExample_ref(self,prev2Tracer,prevTracer,tracer,interId):
        """
        interIdで指定する交差点周りの状況変化（車両）を記録するtf.Exampleを作成する
        """
        signalState=tracer.worldSignalState.signalStateDict[interId]
        prevSignalState=prevTracer.worldSignalState.signalStateDict[interId]
        
        prevVehicleNumInbound,prevVehicleSpeedInbound=self._calcVehicleNumAndSpeedInbound(prevTracer,interId)
        prevVehicleNumOutbound,prevVehicleSpeedOutbound=self._calcVehicleNumAndSpeedOutbound(prevTracer,interId)
        
        prev2VehicleNumInbound,prev2VehicleSpeedInbound=self._calcVehicleNumAndSpeedInbound(prev2Tracer,interId)
        prev2VehicleNumOutbound,prev2VehicleSpeedOutbound=self._calcVehicleNumAndSpeedOutbound(prev2Tracer,interId)

        #prev2VehicleAverageTotalRouteLengthInbound=self._calcAverageTotalRouteLengthInbound(prev2Tracer,interId)
        prev2VehicleAverageTravelTimeInbound=self._calcAverageTravelTimeInbound(prev2Tracer,interId)
                
        normalizedRoadLengthInbound=tracer.getRelativeRoadLengthList(interId,self.segmentLength,"inbound")
        normalizedRoadLengthOutbound=tracer.getRelativeRoadLengthList(interId,self.segmentLength,"outbound")

        roadSpeedLimitInbound=tracer.getRoadSpeedLimitList(interId,"inbound")
        roadSpeedLimitOutbound=tracer.getRoadSpeedLimitList(interId,"outbound")
        
        interSignalized=tracer.connectedInterSignalized(interId)
        
        vehicleSumRunDistance1=self._calcSumRunDistance(prev2Tracer,prevTracer,interId,normalize=False)
        vehicleSumRunDistance2=self._calcSumRunDistance(prev2Tracer,tracer,interId,normalize=False)
        
        assert prevSignalState.phase==signalState.prevPhase,"{}: {}!={}".format(
            interId,
            prevSignalState.phase,
            signalState.phase,
        )
                
        feature = {
          'oldPhase': _int64_feature(prevSignalState.prevPhase),
          'action': _int64_feature(prevSignalState.phase),
          'nextAction': _int64_feature(signalState.phase),
          'interSignalized': _int64_feature(interSignalized),
          'normalizedRoadLengthInbound': _float_feature(normalizedRoadLengthInbound),
          'normalizedRoadLengthOutbound': _float_feature(normalizedRoadLengthOutbound),
          'roadSpeedLimitInbound': _float_feature(roadSpeedLimitInbound),
          'roadSpeedLimitOutbound': _float_feature(roadSpeedLimitOutbound),
          'vehicleNumInbound': _int64_feature(prev2VehicleNumInbound),
          'vehicleSpeedInbound': _float_feature(prev2VehicleSpeedInbound),
          'vehicleNumOutbound': _int64_feature(prev2VehicleNumOutbound),
          'vehicleSpeedOutbound': _float_feature(prev2VehicleSpeedOutbound),
          'nextVehicleNumInbound': _int64_feature(prevVehicleNumInbound),
          'nextVehicleSpeedInbound': _float_feature(prevVehicleSpeedInbound),
          'nextVehicleNumOutbound': _int64_feature(prevVehicleNumOutbound),
          'nextVehicleSpeedOutbound': _float_feature(prevVehicleSpeedOutbound),
          'vehicleSumRunDistance1': _float_feature(vehicleSumRunDistance1),
          'vehicleSumRunDistance2': _float_feature(vehicleSumRunDistance2),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return exampletf.train.Example
    
    def _createExample(self, prevTracer, tracer, interID):
        """
        interIdで指定する交差点周りの状況変化（車両）を記録するtf.Exampleを作成する
        """
        # 交差点の変動情報収集
        nextPhase = tracer.worldSignalState.signalStateDict[interID]
        startPhase = [0]*8 # 0:信号無し, 1-8:phase
        endPhase = [0]*8
        for idx, roadID in enumerate(self.inter2roads['agents'][interID]):
            if roadID == -1:
                continue
            startInterID = self.roadNet.roadDataSet.roadDict[roadID].getStartInterId()
            if startInterID in self.inter2roads['agents'].keys():
                startPhase[idx] = prevTracer.worldSignalState.signalStateDict[startInterID]
            endInterID = self.roadNet.roadDataSet.roadDict[roadID].getEndInterId()
            if endInterID in self.inter2roads['agents'].keys():
                endPhase[idx] = prevTracer.worldSignalState.signalStateDict[endInterID]
            
        def one_hot(raw, dim):
            return [[1 if i == v else 0 for i in range(dim)] for v in raw]
        
        def flatten(list_of_list):
            return list(itertools.chain.from_iterable(list_of_list))
        
        def aggregate(key):
            return [self.road2vehicles[roadID][key] for roadID in self.inter2roads['agents'][interID]]

        feature = {
            'startPhase': _int64_feature(flatten(one_hot(startPhase, 9))),           # road1~8 x 9dim one-hot
            'endPhase': _int64_feature(flatten(one_hot(endPhase, 9))),             # road1~8 x 9dim one-hot
            'startIntersectionPos': _int64_feature(flatten(one_hot(self.agentStaticInfo[interID]['startIntersectionPos'], 5))), # road1~8 x 5dim one-hot
            'endIntersectionPos': _int64_feature(flatten(one_hot(self.agentStaticInfo[interID]['endIntersectionPos'], 5))),   # road1~8 x 5dim one-hot
            'speedLimit': _float_feature(self.agentStaticInfo[interID]['speedLimit']),           # road1~8 (x float)
            'permissibleMove': _int64_feature(flatten(self.agentStaticInfo[interID]['permissibleMove'])),      # lane1~24 x 3dim
            'stackingVehicles': _int64_feature(flatten(aggregate('stackingVehicles'))),     # lane1~24 x 1
            'stackingVehiclesSpeed': _float_feature(flatten(aggregate('stackingVehiclesSpeed'))),     # lane1~24 x 1
            'vehiclesInfo': _float_feature(flatten(flatten(flatten(aggregate('vehiclesInfo'))))),         # lane1~24 x ceil(max_len/5) x 4
            'nextPhase': _int64_feature(flatten(one_hot([nextPhase], 9))),            # 9dim one-hot
            'totalTravelDistance': _float_feature(sum(aggregate('totalTravelDistance')))  # (float)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example
    
    def _createStaticExample(self):
        cardDir = ['N', 'E', 'S', 'W', 'undefined']
        def _replaceNone(value):
            return value.roadId if value is not None else -1
        
        # 隣接道路の接続情報収集
        self.inter2roads = {'agents':{}, 'non-agents':{}}
        # inter2roads['agents'][interID]     = [No, Ni, Eo, Ei, So, Si, Wo, Wi]
        # inter2roads['non-agents'][interID] = [Xo, Xi, Xo, Xi, Xo, Xi, Xo, Xi]
        
        for interID, interObj in self.roadNet.intersectionDataSet.intersectionDict.items():
            if interObj.signal:
                signalObj = self.roadNet.intersectionDataSet.signalDict[interID]
                interRoadDict = [signalObj.outRoadDict, signalObj.inRoadDict]
                self.inter2roads['agents'][interID] = [_replaceNone(interRoadDict[i%2][cardDir[int(i//2)]]) for i in range(8)]
            else:
                self.inter2roads['non-agents'][interID] = [-1]*8
        
        max_length = 0.0
        for roadID, roadObj in self.roadNet.roadDataSet.roadDict.items():
            roadDict = roadObj.exportAsDict()
            # update max
            max_length = max(max_length, roadObj.roadSegment.length)
            self.max_laneNum = max(self.max_laneNum, roadObj.getNumLane())
            
            # update 'non-agents' inter2roads info
            for i, key in enumerate(['startInterId', 'endInterId']):
                interID = roadDict[key]
                if interID in self.inter2roads['non-agents']:
                    ioRoads = self.inter2roads['non-agents'][interID][i::2]
                    self.inter2roads['non-agents'][interID][ioRoads.index(-1)*2+i] = roadID
        
        self.sectionNum = int(-((-max_length)//self.sectionLen))

        # 隣接道路の固定情報収集
        self.agentStaticInfo = {}
        for interID, interObj in self.roadNet.intersectionDataSet.signalDict.items():
            interPos = {'start':[4]*8, 'end':[4]*8} #0-3:NESW, 4:undefined
            speedLimit = [0.0]*8
            permissibleMove = []
            for idx, roadID in enumerate(self.inter2roads['agents'][interID]):
                if roadID == -1:
                    permissibleMove.append([0, 0, 0]*self.max_laneNum)
                else:
                    roadDict = self.roadNet.roadDataSet.roadDict[roadID].exportAsDict()
                    for edgeKey, edgeInterKey in [('start', 'startInterId'), ('end', 'endInterId')]:
                        edgeInterID = roadDict[edgeInterKey]
                        if edgeInterID in self.inter2roads['agents']:
                            interPos[edgeKey][idx] = int(self.inter2roads['agents'][edgeInterID].index(roadID)//2)
                    speedLimit[idx] = roadDict['speedLimit']
                    permissibleMove.append(roadDict['permissibleMovementList'])
            
            self.agentStaticInfo[interID] = {
                'startIntersectionPos': interPos['start'],
                'endIntersectionPos': interPos['end'],
                'speedLimit': speedLimit,
                'permissibleMove': permissibleMove,
            }
            
        return
    
    def _createExampleInit(self, prevTracer, tracer):
        # 車両の変動情報の変換 (TODO:実行場所変える, インターごとでは無駄)
        self.road2vehicles = {
            roadID : {
                 'vehiclesInfo': [
                     [
                         [min(max(roadObj.roadSegment.length-i*self.sectionLen, 0.0)/self.sectionLen, 1.0), 0, 0.0, 0.0] for i in range(self.sectionNum)
                     ] for _ in range(self.max_laneNum)
                 ],
                 'stackingVehicles': [0]*self.max_laneNum,
                 'stackingVehiclesSpeed': [0.0]*self.max_laneNum,
                 'totalTravelDistance': 0.0
                }
            for roadID, roadObj in self.roadNet.roadDataSet.roadDict.items()
        }
                                               
        def _getInterOutRoads(interID):
            return [r for r in (self.inter2roads['agents'].get(interID, self.inter2roads['non-agents'].get(interID)))[::2] if r != -1]
        
        for vehicleID, vehicleObj in prevTracer.vehicleDS.existingVehicleDict.items():
            if math.isclose(vehicleObj.distance, 0.0):
                self.road2vehicles[vehicleObj.roadId]['stackingVehicles'][vehicleObj.lane] += 1
                self.road2vehicles[vehicleObj.roadId]['stackingVehiclesSpeed'][vehicleObj.lane] += vehicleObj.speed
            else:
                sectionNo = int(vehicleObj.distance//self.sectionLen)
                sectionDistance = vehicleObj.distance%self.sectionLen
                if math.isclose(sectionDistance, 0.0):
                    sectionNo -= 1
                sectionInfo = self.road2vehicles[vehicleObj.roadId]['vehiclesInfo'][vehicleObj.lane][sectionNo]
                assert sectionInfo[1] == 0, "overlapping vehicles: {}, {}".format(
                    (sectionNo + sectionInfo[2]*sectionInfo[0]) * self.sectionLen,
                    vehicleObj.distance
                )
                sectionInfo[1] = 1
                if math.isclose(sectionInfo[0], 0.0):
                    print(vehicleObj.distance)
                    print(self.roadNet.roadDataSet.roadDict[vehicleObj.roadId].roadSegment.length)
                sectionInfo[2] = sectionDistance / (sectionInfo[0]*self.sectionLen)
                sectionInfo[3] = vehicleObj.speed
                
            roadObj = self.roadNet.roadDataSet.roadDict[vehicleObj.roadId]
            vehicleObjFuture = tracer.vehicleDS.allVehicleDict[vehicleID]
            if vehicleObj.roadId == vehicleObjFuture.roadId:
                self.road2vehicles[vehicleObj.roadId]['totalTravelDistance'] += vehicleObjFuture.distance - vehicleObj.distance
            elif vehicleObjFuture.roadId in _getInterOutRoads(roadObj.getEndInterId()):
                self.road2vehicles[vehicleObj.roadId]['totalTravelDistance'] += (vehicleObjFuture.distance 
                                                                                 + roadObj.roadSegment.length - vehicleObj.distance)
            else:
                travelDistance = 0
                for passedRoadID in _getInterOutRoads(roadObj.getEndInterId()):
                    passedRoadObj = self.roadNet.roadDataSet.roadDict[passedRoadID]
                    if vehicleObjFuture.roadId in _getInterOutRoads(passedRoadObj.getEndInterId()):
                        travelDistance = vehicleObjFuture.distance + passedRoadObj.roadSegment.length + roadObj.roadSegment.length - vehicleObj.distance
                        break
                assert travelDistance > 0, "+2 hop distanace"
                self.road2vehicles[vehicleObj.roadId]['totalTravelDistance'] += travelDistance
                
        self.road2vehicles[-1] = {
             'vehiclesInfo': [
                 [
                     [0.0, 0, 0.0, 0.0] for i in range(self.sectionNum)
                 ] for _ in range(self.max_laneNum)
             ],
             'stackingVehicles': [0]*self.max_laneNum,
             'stackingVehiclesSpeed': [0.0]*self.max_laneNum,
             'totalTravelDistance': 0.0
        }
            
        return
    
    def createTFRecord(self,tfrecordPath):
        with tf.io.TFRecordWriter(tfrecordPath) as writer:
            for example in self.exampleList:
                writer.write(example.SerializeToString())
        print("tfrecord created:",tfrecordPath)
                
    def _action(self, observations):
        return self.strategyRunner.getActionsFunc(observations)
    
    def decideActions(self,
            observations,
            prevActCountInEpisode,
            runType,
            exitType=False,
            debug=False
        ):
        collect=(prevActCountInEpisode>=1)
        print(collect)
            
        if collect:
            tracer=self._createRoadTracer(observations)
            prevTracer=self._createPrevRoadTracer()
            self._createExampleInit(prevTracer, tracer)
        
        print("before action")
        actions=self._action(observations) #need call even if step is LAST for saving last action for next episode
        print("after action")
        for interId,signalState in self.worldSignalState.signalStateDict.items():
            if collect:
                example=self._createExample(prevTracer,tracer,interId)
                self.exampleList.append(example)
            
            select_phase=actions[interId] if interId in actions else signalState.phase
            
            signalState.setPreviousStepInfo()
            if select_phase != signalState.phase:
                signalState.changePhase(select_phase,observations.current_time)
            else:
                signalState.keepPhase(observations.current_time)
        print("end action")
        return actions
    
def list_slice(x,indices):
    return tf.stack([x[:,i,:] for i in indices], axis=1) 

def _createDenceList(name,denseLayerUnits):
    return [
        tf.keras.layers.Dense(
            name='{}_{}'.format(name,idx),
            units=units,
            activation='relu'
        )
        for idx,units in enumerate(denseLayerUnits)
    ]
    
class EmbeddingLayer(tf.keras.layers.Layer):
    # あるタイミングでの道路（inbound,outboundの両方）上の車両統計情報と信号フェーズ情報を入力すると、
    # そのタイミングでの「状況」を表す特徴ベクトルに変換（embedding）するレイヤー。
    # 特徴ベクトルは、アクション（次に選択する信号フェーズ）毎に算出され、
    # アクションに応じて次ステップにおいてoutboundな道路における車両統計情報がどうなるかを予測するのに十分な情報を含みます。
    # 
    # 記号の意味:
    #     RunDistanceModelクラスのコメントを参照
    # inputの仕様
    # 　shape: (None,24*Di+8*Do+21)
    # 　input[any,:]の値の定義: 下記の順序で要素が並べたもの
    #     4個の要素 =  (Sg_0,Sg_1,Sg_2,Sg_3)
    #     12*Di個の要素 =  (Ni_0_0,...,Ni_11_0,Ni_0_1,...,Ni_11_1,...,Ni_0_Di-1,...,Ni_11_Di-1)
    #     12*Di個の要素 =  (Si_0_0,...,Si_11_0,Si_0_1,...,Si_11_1,...,Si_0_Di-1,...,Si_11_Di-1)
    #     4個の要素 =  (Li_0,Li_1,Li_2,Li_3)
    #     4個の要素 =  (Mi_0,Mi_1,Mi_2,Mi_3)
    #     4*Do個の要素 =  (No_0_0,...,No_3_0,No_0_1,...,No_3_1,...,No_0_Do-1,...,No_3_Do-1)
    #     4*Do個の要素 =  (So_0_0,...,So_3_0,So_0_1,...,So_3_1,...,So_0_Do-1,...,So_3_Do-1)
    #     4個の要素 =  (Lo_0,Lo_1,Lo_2,Lo_3)
    #     4個の要素 =  (Mo_0,Mo_1,Mo_2,Mo_3)
    #     1個の要素 =  Pprev
    # outputの仕様
    # 　4つのtensorからなるtuple(inflowEmbedding,outboundFuture,estimatedSumRunDistance,estimatedSumRunDistance2)を返します。
    #   inflowEmbedding: 現在inboundな道路上にいる車両が、アクション（選択する信号フェーズ）に応じて、outboundな道路に進行した未来状態を表す特徴ベクトル
    #      shape: (None,4,8,Dv)
    #      inflowEmbedding[any,direction,phase,:]の値は下記条件での特徴ベクトルを表す
    #          direction: outboundな道路の方角(N,E,S,W) 
    #          phase: アクション（選択する信号フェーズ）
    #   outboundFuture: outboundな道路の未来状態を表す特徴ベクトル
    #                   ＝現在outboundな道路上にいる車両が進行した未来状態と、inflowEmbeddingが表す未来状態の両方を含む特徴ベクトル
    #      shape: (None,4,8,Dv)
    #      outboundFuture[any,direction,phase,:]の値は下記条件での特徴ベクトルを表す
    #          direction: outboundな道路の方角(N,E,S,W) 
    #          phase: アクション（選択する信号フェーズ）
    #   estimatedSumRunDistance: アクション（選択する信号フェーズ）毎に１ステップの間にinboundな道路上の車両が進む距離の合計の予測値
    #      shape: (None,8,Dv)
    #      estimatedSumRunDistance[any,phase,:]の値は下記条件での特徴ベクトルを表す
    #          phase: アクション（選択する信号フェーズ）
    #   estimatedSumRunDistance2: estimatedSumRunDistanceと同様だが、1ステップでなく2ステップとした予測値

    #抜ける方角(N,E,S,W)の順に許可する進行方向を定義（0:左折と右折を許可,1:直進と右折を許可,2:右折を許可）
    phaseToLTRindexDict={
        1:[2,0,2,0], #phase1は「左折して東に抜ける」「左折して西に抜ける」も許可
        2:[1,2,1,2], #phase2は「直進して北に抜ける」「直進して南に抜ける」も許可
        3:[0,2,0,2], #phase3は「左折して北に抜ける」「左折して南に抜ける」も許可
        4:[2,1,2,1], #phase4は「直進して東に抜ける」「直進して西に抜ける」も許可
        5:[2,0,1,2],
        6:[2,2,0,1],
        7:[1,2,2,0],
        8:[0,1,2,2],
    }
    defaultUnitsDict={
        'inbound':[64,64,64],
        'outbound':[64,64,64],
        'inflow':[64,64,64],
        'runDistance':[64,64,64],
    }
    outputUnitsDict={ #None means that output units is dimVehiclesVec
        'inbound': None,
        'outbound':None,
        'inflow':None,
        'runDistance':1,
    }
    
    def __init__(self,
            numSegmentInbound,
            numSegmentOutbound,
            *,
            unitsDict={},
            dimVehiclesVec=64,
            version=1,
            name=None
        ):
        super().__init__(name=name)
        self.version=version
        
        denseLayerUnitsDict={}
        for name,defaultUnitsList in self.defaultUnitsDict.items():
            if name not in unitsDict or unitsDict[name] is None:
                denseLayerUnitsDict[name]=copy.copy(defaultUnitsList)
            else:
                denseLayerUnitsDict[name]=copy.copy(unitsDict[name])
            
            outputUnits = self.outputUnitsDict[name]
            if outputUnits is None:
                outputUnits=dimVehiclesVec
            denseLayerUnitsDict[name].append(outputUnits)
        
        self.numSegmentInbound=numSegmentInbound
        self.numSegmentOutbound=numSegmentOutbound
        self.dimVehiclesVec=dimVehiclesVec
        
        # inboundなレーン上の車両の情報を特徴ベクトル化（inbound特徴ベクトル）する層
        self.denseInboundList = _createDenceList("inbound",denseLayerUnitsDict["inbound"])
        # outboundな道路上の車両の情報を特徴ベクトル化（outbound特徴ベクトル）する層
        self.denseOutboundList = _createDenceList("outbound",denseLayerUnitsDict["outbound"])
        # inbound特徴ベクトルとoutbound特徴ベクトルを加算した情報を処理してinboundな車両がoutboundな道路に与える影響（inflowベクトル）を求める層
        self.denseInflowList = _createDenceList("inflow",denseLayerUnitsDict["inflow"])
        
        self.densePhaseMask = tf.keras.layers.Dense(
            name='phase_mask',
            units=dimVehiclesVec,
            activation='sigmoid',
            use_bias=False
        )

        self.denseRunDistanceBeforeInterList = _createDenceList("rundist_before",denseLayerUnitsDict["runDistance"])
        self.denseRunDistanceAfterInterList = _createDenceList("rundist_after",denseLayerUnitsDict["runDistance"])
        self.denseRunDistance2BeforeInterList = _createDenceList("rundist2_before",denseLayerUnitsDict["runDistance"])
        self.denseRunDistance2AfterInterList = _createDenceList("rundist2_after",denseLayerUnitsDict["runDistance"])
        
        '''
        self.phase_mask = self.add_weight(
            shape=(8,),
            initializer="random_normal",
            trainable=True,
        )
        '''
        
    def getInputDim(self):
        return 24*self.numSegmentInbound+8*self.numSegmentOutbound+21    

    @staticmethod
    def calcSumRunDistance(
            inboundVehicles,
            inflowEmbedding,
            layersBeforeInterList,
            layersAfterInterList,
        ):
        #Ri:交差点の信号と独立に決まるsumRunDistanceの成分
        #Ro:交差点の信号によって決まるsumRunDistanceの成分
        
        Ri=inboundVehicles# shape = [BATCH,12,Dv]
        for layer in layersBeforeInterList:
            Ri = layer(Ri) # shape = [BATCH,12,1]
        Ri = tf.reduce_sum(Ri,axis=1) # shape = [BATCH,1]
        Ri = tf.expand_dims(Ri,axis=1) # shape = [BATCH,1,1]
        
        Ro=inflowEmbedding# shape = [BATCH,4,8,Dv]
        for layer in layersAfterInterList:
            Ro = layer(Ro) # shape = [BATCH,4,8,1]
        Ro = tf.reduce_sum(Ro,axis=1) # shape = [BATCH,8,1]
        
        sumRunDistance=Ri+Ro
        return sumRunDistance

    @staticmethod
    def _sliceByList(inputs,numList):
        slicedList=[]
        
        sumNum=0
        for num in numList:
            slicedList.append(inputs[:,sumNum:sumNum+num])
            sumNum+=num
        return slicedList
    
    @tf.function
    def call(self, inputs):
        print('Tracing EmbeddingLayer')
        Di=self.numSegmentInbound
        Do=self.numSegmentOutbound
        Dv=self.dimVehiclesVec

        ################## 1. 入力を分解する
        _Sg,_Ni,_Si,_Li,_Mi,_No,_So,_Lo,_Mo,Pprev = self._sliceByList(inputs,[4,12*Di,12*Di,4,4,4*Do,4*Do,4,4,1])
        Pprev=tf.cast(Pprev[:,0],tf.uint8)
        
        Sg=tf.reshape(tf.repeat(_Sg,repeats=3),[-1,12,1]) # shape = [BATCH,12,1]
        Ni=tf.transpose(tf.reshape(_Ni,[-1,Di,12]),perm=[0, 2, 1]) # shape = [BATCH,12,Di]
        Si=tf.transpose(tf.reshape(_Si,[-1,Di,12]),perm=[0, 2, 1]) # shape = [BATCH,12,Di]
        Li=tf.reshape(tf.repeat(_Li,repeats=3),[-1,12,1]) # shape = [BATCH,12,1]
        Mi=tf.reshape(tf.repeat(_Mi,repeats=3),[-1,12,1]) # shape = [BATCH,12,1]
        No=tf.transpose(tf.reshape(_No,[-1,Do,4]),perm=[0, 2, 1]) # shape = [BATCH,4,Do]
        So=tf.transpose(tf.reshape(_So,[-1,Do,4]),perm=[0, 2, 1]) # shape = [BATCH,4,Do]
        Lo=tf.reshape(_Lo,[-1,4,1]) # shape = [BATCH,4,1]
        Mo=tf.reshape(_Mo,[-1,4,1]) # shape = [BATCH,4,1]
        
        ################## 2. inboundなレーン上の車両を表現する特徴ベクトル（inbound）を得る
        inboundVehicles = tf.concat([Sg,Ni,Si,Li,Mi],axis=2) # shape = [BATCH,12,2*Di+3]
        for layer in self.denseInboundList:
            inboundVehicles = layer(inboundVehicles)
        #inboundVehicles : [BATCH,12,Dv]
        
        ################## 3. outboundな道路上の車両を表現する特徴ベクトル（outbound）を得る
        outboundVehicles = tf.concat([No,So,Lo,Mo],axis=2) # shape = [BATCH,4,2*Do+2]
        for layer in self.denseOutboundList:
            outboundVehicles = layer(outboundVehicles)
        #outboundVehicles : [BATCH,4,Dv] 
            
        ################## 4. 特徴ベクトルをまとめなおしてた上で、inboundとoutboundの情報を統合した特徴ベクトル（inflow）を得る
        #交差点進行後の方向（N,E,S,W）別にまとめる。各まとまり中では進行方向が(L,T,R)の順に並べる
        _inboundVehicles_group=tf.stack([
                list_slice(inboundVehicles,[9,7,5]), # shape = [BATCH,3,Dv]
                list_slice(inboundVehicles,[0,10,8]), # shape = [BATCH,3,Dv]
                list_slice(inboundVehicles,[3,1,11]), # shape = [BATCH,3,Dv]
                list_slice(inboundVehicles,[6,4,2]), # shape = [BATCH,3,Dv]
            ],axis=1
        ) # shape = [BATCH,4,3,Dv]
        inboundVehicles_group=tf.stack([
                _inboundVehicles_group[:,:,0,:]+_inboundVehicles_group[:,:,2,:], #右折の情報を左折の情報に加算する
                _inboundVehicles_group[:,:,1,:]+_inboundVehicles_group[:,:,2,:], #右折の情報を直進の情報に加算する
                _inboundVehicles_group[:,:,2,:]
        ],axis=2
        ) # shape = [BATCH,4,3,Dv]
        
        outboundVehicles_groupe=tf.expand_dims(outboundVehicles,axis=2)# shape = [BATCH,4,1,Dv]
        
        inflowVehicles=tf.concat([
            inboundVehicles_group, # shape = [BATCH,4,3,Dv]
            tf.tile(outboundVehicles_groupe,[1,1,3,1])# shape = [BATCH,4,3,Dv]
        ],axis=3) # shape = [BATCH,4,3,2*Dv]
        
        for layer in self.denseInflowList:
            inflowVehicles = layer(inflowVehicles)
        #inflowVehicles: [BATCH,4,3,Dv]
        
        ################## 5. 特徴ベクトル（inflow）をphase毎にまとめなおした上で現在のphaseで決まるmaskを乗算し、特徴ベクトル（inflowEmbedding）を得る
        inflowVehiclesByPhaseList=[None]*8
        for phase_zero_base in range(8):
            inflowVehiclesByPhaseList[phase_zero_base]=tf.stack([
                inflowVehicles[:,i,j,:] for i,j in enumerate(self.phaseToLTRindexDict[phase_zero_base+1])  # shape = [BATCH,Dv]
            ],axis=1)# shape = [BATCH,4,Dv]
        inflowVehiclesByPhase=tf.stack(inflowVehiclesByPhaseList,axis=2)# shape = [BATCH,4,8,Dv]
        
        Pprev_one_hot_reverse=tf.one_hot(Pprev-1,depth=8,on_value=0.,off_value=1.,dtype=tf.float32) # shape = [BATCH,8]
        _mask_reverse=self.densePhaseMask(tf.constant([[1]])) # shape = [1,Dv]
        _mask_reverse=tf.reshape(_mask_reverse,[Dv])# shape = [Dv]
        mask=1-tf.tensordot(Pprev_one_hot_reverse,_mask_reverse,axes=0) # shape = [BATCH,8,Dv]
        mask=tf.expand_dims(mask,axis=1) # shape = [BATCH,1,8,Dv]
        
        inflowEmbedding = inflowVehiclesByPhase*mask # shape = [BATCH,4,8,Dv]
        
        ################## 6. 特徴ベクトル（inflowEmbedding）と特徴ベクトル（outbound）を加算（skip接続）
        # この接続によりinflowEmbeddingは「outbound情報のみで決まる特徴」を含まない
        #「inboud情報・outbound情報・phase情報によって決まる特徴」を表すようになる
        _outboundVehicles=tf.expand_dims(outboundVehicles,axis=2) # shape = [BATCH,4,1,Dv]
        outboundFuture=_outboundVehicles+inflowEmbedding # shape = [BATCH,4,8,Dv]
        
        ################## 7. 特徴ベクトル（inflowEmbedding）と特徴ベクトル（inbound）からinboundな車両の走行距離を予測する
        estimatedSumRunDistance=self.calcSumRunDistance(
            inboundVehicles,
            inflowEmbedding,
            self.denseRunDistanceBeforeInterList,
            self.denseRunDistanceAfterInterList,
        )
        estimatedSumRunDistance2=self.calcSumRunDistance(
            inboundVehicles,
            inflowEmbedding,
            self.denseRunDistance2BeforeInterList,
            self.denseRunDistance2AfterInterList,
        )
        ###############
        
        return inflowEmbedding,outboundFuture,estimatedSumRunDistance,estimatedSumRunDistance2

    
class EmbeddingSimpleLayer(EmbeddingLayer):
    def __init__(self,
            *args,
            **kwargs
        ):
        super().__init__(*args,**kwargs)
    def call(self, inputs):
        inflowEmbedding,_,_,_ = super().call(inputs)
        return inflowEmbedding
    
class RunDistanceModel_ref(tf.keras.Model):
    # 車両情報と信号フェーズ情報を用いて、未来状態を予測するタスクを解くネットワーク。
    # ※上記タスクを解くことで、車両情報と信号フェーズ情報を特徴ベクトル化（embedding）するネットワークが内部につくられる 
    #
    # 記号の意味:
    #     Di = （定数）交差点へのinboundな道路について交差点に近い位置から何セグメントまで考慮するかの値
    #     Do = （定数）交差点からのoutboundな道路について交差点に近い位置から何セグメントまで考慮するかの値
    #           ※セグメントとは：道路を一定距離（例：25m）で切断してえられる部分区間
    #
    #     [車両に関する情報]
    #     Ni_x_y = 1ステップ前の時刻における、交差点へのinboundな4本の道路におけるx番目のレーンのy番目のセグメント上の車両の数
    #     Si_x_y = 1ステップ前の時刻における、交差点へのinboundな4本の道路におけるx番目のレーンのy番目のセグメント上の車両の速度の合計値
    #     No_x_y = 1ステップ前の時刻における、交差点からのoutboundな4本の道路におけるx番目の道路のy番目のセグメント上の車両の数
    #     So_x_y = 1ステップ前の時刻における、交差点からのoutboundな4本の道路におけるx番目の道路のy番目のセグメント上の車両の速度の合計値
    #     No'_x_y = 現時刻における、交差点からのoutboundな4本の道路におけるx番目の道路のy番目のセグメント上の車両の数
    #     So'_x_y = 現時刻における、交差点からのoutboundな4本の道路におけるx番目の道路のy番目のセグメント上の車両の速度の合計値
    #
    #     [道路構造に関する情報]　※時刻によって変わらない
    #     Sg_x = 交差点に接続する4本の道路の先の交差点の信号の有無
    #     Li_x = 交差点へのinboundな4本の道路におけるx番目の道路の正規化した長さ（正規化＝セグメント長を１とした長さ）
    #     Lo_x = 交差点からのoutboundな4本の道路におけるx番目の道路の正規化した長さ（正規化＝セグメント長を１とした長さ）
    #     Mi_x = 交差点へのinboundな4本の道路におけるx番目の道路の制限速度
    #     Mo_x = 交差点からのoutboundな4本の道路におけるx番目の道路の制限速度
    #
    #     [信号のphaseに関する情報]　※時刻によって変わる
    #     Pprev = 1ステップ前の時刻におけるアクション前の信号フェーズ（＝2ステップ前の時刻から1ステップ前の時刻における信号フェーズ）
    #     Pnow = 1ステップ前の時刻におけるアクション後の信号フェーズ（＝1ステップ前の時刻から現時刻における信号フェーズ）
    #
    #     [学習に使用するmask情報]　※時刻によって変わる
    #     Ak = sumRunDistance2のlossを計算する場合は1,計算しない場合は0
    #
    #     [補足]
    #     道路の番号は、方角(N,E,S,W)の順番で採番（0ベース）
    #     レーンの番号は、方角(N,E,S,W)と進行方向（L,T,R）の組み合わせについて下記の順番で採番（0ベース）
    #       (N,L),(N,T),(N,R),(E,L),(E,T),(E,R),(S,L),(S,T),(S,R),(W,L),(W,T),(W,R)
    #     セグメントの番号は、等距離に分割された区間について交差点に近いものから順に採番（0ベース）
    #
    # 冗長性：
    # 　　今回の問題設定では、Li_x==Lo_x, Mi_x==Mo_xが常にな成り立つので情報として分ける必要は本来ない。
    #
    # inputの仕様
    # 　shape: (None,24*Di+8*Do+23)
    # 　input[any,:]の値の定義: 下記の順序で要素が並べたもの
    #     4個の要素 =  (Sg_0,Sg_1,Sg_2,Sg_3)
    #     12*Di個の要素 =  (Ni_0_0,...,Ni_11_0,Ni_0_1,...,Ni_11_1,...,Ni_0_Di-1,...,Ni_11_Di-1)
    #     12*Di個の要素 =  (Si_0_0,...,Si_11_0,Si_0_1,...,Si_11_1,...,Si_0_Di-1,...,Si_11_Di-1)
    #     4個の要素 =  (Li_0,Li_1,Li_2,Li_3)
    #     4個の要素 =  (Mi_0,Mi_1,Mi_2,Mi_3)
    #     4*Do個の要素 =  (No_0_0,...,No_3_0,No_0_1,...,No_3_1,...,No_0_Do-1,...,No_3_Do-1)
    #     4*Do個の要素 =  (So_0_0,...,So_3_0,So_0_1,...,So_3_1,...,So_0_Do-1,...,So_3_Do-1)
    #     4個の要素 =  (Lo_0,Lo_1,Lo_2,Lo_3)
    #     4個の要素 =  (Mo_0,Mo_1,Mo_2,Mo_3)
    #     1個の要素 =  Pprev
    #     1個の要素 =  Pnow
    #     1個の要素 =  Ak
    #
    # outputの仕様
    # 　shape: (None,8*Do+2)
    # 　output[any,:]の値の定義: 下記の順序で要素が並べたもの
    #     4*Do個の要素 =  (No'_x_y,...,No'_3_0,No'_0_1,...,No'_3_1,...,No'_0_Do-1,...,No'_3_Do-1)
    #     4*Do個の要素 =  (So'_0_0,...,So'_3_0,So'_0_1,...,So'_3_1,...,So'_0_Do-1,...,So'_3_Do-1)
    #     1個の要素 =  sumRunDistance
    #     1個の要素 =  sumRunDistance2

    def __init__(self,
            numSegmentInbound,
            numSegmentOutbound,
            unitsDict={},
            dimVehiclesVec=64,
            version=1,
        ):
        super().__init__()
        self.version=version
        self.embeddingLayer=EmbeddingLayer(
            numSegmentInbound,
            numSegmentOutbound,
            unitsDict=unitsDict,
            dimVehiclesVec=dimVehiclesVec,
            version=version,
            name='embedding',
        )
        
        if "output" not in unitsDict or unitsDict["output"] is None:
            denseLayerUnitsForOutput=[]
        else:
            denseLayerUnitsForOutput=copy.copy(unitsDict["output"])
            
        denseLayerUnitsForOutput.append(2*numSegmentOutbound)
        self.denseOutputList = _createDenceList("output",denseLayerUnitsForOutput)
        
    def getEmbeddedVehiclesVecLength(self):
        return self.embeddingLayer.dimVehiclesVec
    
    def getInputDim(self):
        return 24*self.embeddingLayer.numSegmentInbound+8*self.embeddingLayer.numSegmentOutbound+23    

    def calcEmbeddingVec(self,tracer,interId,segmentLength):
        inputCreator=InputArrayCreator(
            self.embeddingLayer.numSegmentInbound,
            self.embeddingLayer.numSegmentOutbound,
            segmentLength,
        )
        inputs = inputCreator.createOneSignalInputForEmbeddingLayer(tracer,interId)
        inputs=tf.convert_to_tensor(inputs)
        _embeddedVecBatch,_,_eSumRunDistBatch,_eSumRunDist2Batch = self.embeddingLayer(inputs[np.newaxis,:])
        phaseProb = tf.nn.softmax(_eSumRunDist2Batch[0,:,:],axis=0) #shape = [8,1]
        
        embeddedVecAveraged=tf.matmul(
            tf.tile(tf.expand_dims(phaseProb,axis=0),[4,1,1]), # shape = [4,8,1]
            _embeddedVecBatch[0,:,:,:], #shape = [4,8,Dv] 
            transpose_a=True
        ) # shape = [4,1,Dv]
        output=embeddedVecAveraged[:,0,:]# shape = [4,Dv]
        return output.numpy(), phaseProb[:,0].numpy()
    
    def calcEmbeddingVecAll(self,tracer,segmentLength):
        inputCreator=InputArrayCreator(
            self.embeddingLayer.numSegmentInbound,
            self.embeddingLayer.numSegmentOutbound,
            segmentLength,
        )
        inputs,interIds = inputCreator.createAllSignalInputForEmbeddingLayer(tracer)
        inputs=tf.convert_to_tensor(inputs)
        BATCH=64
        outputList=[]
        phaseProbList=[]
        for i in range(math.ceil(len(interIds)/BATCH)):
            _embeddedVecBatch,_,_eSumRunDistBatch,_eSumRunDist2Batch = self.embeddingLayer(inputs[i*BATCH:(i+1)*BATCH,:])
            phaseProb = tf.nn.softmax(_eSumRunDist2Batch,axis=1) #shape = [BATCH,8,1]
            embeddedVecAveraged=tf.matmul(
                tf.tile(tf.expand_dims(phaseProb,axis=1),[1,4,1,1]), # shape = [BATCH,4,8,1]
                _embeddedVecBatch, #shape = [BATCH,4,8,Dv] 
                transpose_a=True
            ) # shape = [BATCH,4,1,Dv]
            output=embeddedVecAveraged[:,:,0,:]# shape = [BATCH,4,Dv]
            outputList.append(output)
            phaseProbList.append(phaseProb[:,:,0])
        
        outputAll=tf.concat(outputList,axis=0)# shape = [NumInterId,4,Dv]
        phaseProbAll=tf.concat(phaseProbList,axis=0)# shape = [NumInterId,8]
        
        return outputAll.numpy(), phaseProbAll.numpy(),interIds
    
    @tf.function
    def call(self, inputs, training=None):
        print('Tracing RunDistanceModel')
        Do=self.embeddingLayer.numSegmentOutbound
        Dv=self.embeddingLayer.dimVehiclesVec
        
        inputsEmbeddingLayer=inputs[:,:-2]
        Pnow=tf.cast(inputs[:,-2],tf.uint8)
        Ak=tf.cast(inputs[:,-1],tf.float32)
            
        _,_outboundFuture,_eSumRunDist,_eSumRunDist2=self.embeddingLayer(inputsEmbeddingLayer)

        Pnow_one_hot=tf.one_hot(Pnow-1,depth=8) # shape = [BATCH,8]
        Pnow_one_hot=tf.expand_dims(Pnow_one_hot,axis=1) # shape = [BATCH,1,8]

        outboundFuture=tf.matmul(
            tf.tile(tf.expand_dims(Pnow_one_hot,axis=1),[1,4,1,1]), # shape = [BATCH,4,1,8]
            _outboundFuture, # shape = [BATCH,4,8,Dv]
        ) # shape = [BATCH,4,1,Dv]
        
        estimatedVehiclesOnOutboundRoad=tf.reshape(outboundFuture,[-1,4,Dv]) # shape = [BATCH,4,Dv]
        for layer in self.denseOutputList:
            estimatedVehiclesOnOutboundRoad = layer(estimatedVehiclesOnOutboundRoad)
        #estimatedOutboundRoad: [BATCH,4,2*Do]
        
        estimatedVehiclesOnOutboundRoad=tf.transpose(estimatedVehiclesOnOutboundRoad, perm=[0, 2, 1])#shape = [BATCH,2*Do,4]
        estimatedVehiclesOnOutboundRoad=tf.reshape(estimatedVehiclesOnOutboundRoad,[-1,8*Do]) # shape = [BATCH,8*Do]

        ##################
        eSumRunDist=tf.matmul(
            Pnow_one_hot,# shape = [BATCH,1,8]
            _eSumRunDist, # shape = [BATCH,8,1]
        ) # shape = [BATCH,1,1]
        eSumRunDist=tf.reshape(eSumRunDist,[-1,1]) # shape = [BATCH,1]
        ##################
        Ak=tf.reshape(Ak,[-1,1])# shape = [BATCH,1]
        ##################
        eSumRunDist2=tf.matmul(
            Pnow_one_hot,# shape = [BATCH,1,8]
            _eSumRunDist2, # shape = [BATCH,8,1]
        ) # shape = [BATCH,1,1]
        eSumRunDist2=tf.reshape(eSumRunDist2,[-1,1]) # shape = [BATCH,1]
        ##################
        
        outputs=tf.concat([
            estimatedVehiclesOnOutboundRoad,
            eSumRunDist*RUN_DISTANCE_SCALER,
            Ak*eSumRunDist2*RUN_DISTANCE_SCALER
        ],axis=1) #shape = [BATCH,8*Do+1]
        return outputs
    
class RunDistanceModel():
    def __init__(self, sectionNum, embedding_dim = 128):
        self.sectionNum = sectionNum
        self.emb_dim = embedding_dim
    
    def getModel(self):
        roadNum = 8
        laneNum = 3
        lane_detail_shape = (self.sectionNum, 4)
        lane_detail_inputs = [Input(shape=lane_detail_shape, dtype='float32') for _ in range(roadNum*laneNum)]
        
        lane_embedding_layer = LSTM(self.emb_dim, input_shape=lane_detail_shape, return_state=True)
        lane_embeddings = []
        for lane_detail in lane_detail_inputs:
            _, _, state_c = lane_embedding_layer(lane_detail_layer)
            lane_embeddings.append(state_c)
            
        lane_attribute_shape = (5,)
        lane_attribute_inputs = [Input(shape=lane_attribute_shape, dtype='float32') for _ in range(roadNum*laneNum)]
        lane_info = [Concatenate()([em, attr]) for em, attr in zip(lane_embeddings, lane_attribute_inputs)]
        road_detail = [Concatenate()([lane_info[j+i*laneNum] for j in range(laneNum)]) for i in range(roadNum)]
        
        road_embedding_layer1 = Dense(self.emb_dim, input_shape=road_detail[0].shape, activation='tanh')
        road_embeddings = []
        for road_lanes in road_detail:
            road_embeddings.append(road_embedding_layer1(road_lanes))
            
        road_attribute_shape = (29,)
        road_attribute_inputs = [Input(shape=road_attribute_shape, dtype='float32') for _ in range(roadNum)]
        road_info = [Concatenate()([em, attr]) for em, attr in zip(road_embeddings, road_attribute_inputs)]
        road_embeddings = Dense(self.emb_dim//2, input_shape=road_detail[0].shape, activation='tanh')(Reshape((roadNum, -1))(Concatenate()(road_info)))
        out_road_embeddings = [Flatten()(road_embeddings[:,i,:]) for i in range(0, roadNum, 2)]
        in_road_embeddings  = [Flatten()(road_embeddings[:,i,:]) for i in range(1, roadNum, 2)]
        
        out_roads_conv = Dense(self.emb_dim//4, shape=out_road_embeddings[0].shape, activation='tanh')(Average()(out_road_embeddings))
        in_roads_conv  = Dense(self.emb_dim//4, shape= in_road_embeddings[0].shape, activation='tanh')(Average()(in_road_embeddings))
        inter_conv = Activation('tanh')(Add()([out_roads_conv, in_roads_conv]))
        results = Dense(8, shape=inter_conv.shape)(inter_conv)
        
        next_phase_input = Input(shape=(1,), dtype='int64')
        next_phase_mask = tf.one_hot(next_phase_input-1, depth=8)
        result = Dot(axes=0)(results, next_phase_mask)
        
        return keras.Model([lane_detail_inputs, lane_attribute_inputs, road_attribute_inputs, next_phase_input], result)
        