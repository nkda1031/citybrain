import numpy as np

class InputArrayCreator:
    def __init__(self,
            numSegmentInbound,
            numSegmentOutbound,
            segmentLength
        ):
        self.numSegmentInbound=numSegmentInbound
        self.numSegmentOutbound=numSegmentOutbound
        self.segmentLength=segmentLength
        
    def createAllSignalInputForEmbeddingLayer(self,
            tracer,
        ):
        inputArrayList=[]
        interIdList=[]
        for interId in tracer.worldSignalState.signalStateDict:
            inputArray=self.createOneSignalInputForEmbeddingLayer(tracer,interId)
            inputArrayList.append(inputArray)
            interIdList.append(interId)
            
        return np.stack(inputArrayList,axis=0),np.stack(interIdList)
    
    def createOneSignalInputForEmbeddingLayer(self,
            tracer,
            interId,
        ):
        signalState=tracer.worldSignalState.signalStateDict[interId]
        
        vehicleNumInbound,vehicleSpeedInbound=tracer.calcVehicleNumAndSpeedOnSegmentedInboundLane(
            interId,
            self.numSegmentInbound,
            self.segmentLength
        )
        vehicleNumOutbound,vehicleSpeedOutbound=tracer.calcVehicleNumAndSpeedOnSegmentedOutboundRoad(
            interId,
            self.numSegmentOutbound,
            self.segmentLength
        )
        interSignalized=tracer.connectedInterSignalized(interId)
        
        normalizedRoadLengthInbound=tracer.getRelativeRoadLengthList(interId,self.segmentLength,"inbound")
        normalizedRoadLengthOutbound=tracer.getRelativeRoadLengthList(interId,self.segmentLength,"outbound")

        roadSpeedLimitInbound=tracer.getRoadSpeedLimitList(interId,"inbound")
        roadSpeedLimitOutbound=tracer.getRoadSpeedLimitList(interId,"outbound")
        
        inputArray = np.concatenate([
            np.array(interSignalized,dtype=np.float32),
            np.array(vehicleNumInbound,dtype=np.float32),
            np.array(vehicleSpeedInbound,dtype=np.float32),
            np.array(normalizedRoadLengthInbound,dtype=np.float32),
            np.array(roadSpeedLimitInbound,dtype=np.float32),
            np.array(vehicleNumOutbound,dtype=np.float32),
            np.array(vehicleSpeedOutbound,dtype=np.float32),
            np.array(normalizedRoadLengthOutbound,dtype=np.float32),
            np.array(roadSpeedLimitOutbound,dtype=np.float32),
            np.array([signalState.phase],dtype=np.float32),
        ])
            
        return inputArray
