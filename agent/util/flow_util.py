import json

class Flow():
    def __init__(self, start, end, interval, roadIdlist):
        self.start = start
        self.end = end
        self.interval = interval
        self.roadIdlist = roadIdlist

    def getText(self):
        txt = ""
        txt += "{} {} {}\n".format(self.start, self.end, self.interval)
        txt += "{}\n".format(len(self.roadIdlist))
        txt += ' '.join(map(str,self.roadIdlist))+'\n'
        return txt
    
    def exportAsDict(self):
        return {
            'start':self.start,
            'end':self.end,
            'interval':self.interval,
            'roadIdList':self.roadIdlist,
        }

    @classmethod
    def fromText(cls,lines):
        sp0=list(map(int,lines[0].rstrip('n').split(' ')))
        sp2=list(map(int,lines[2].rstrip('n').split(' ')))
        return Flow(sp0[0],sp0[1],sp0[2],sp2)

class FlowDataSet():
    def __init__(self,flowList):
        self.flowList = flowList
    
    def getText(self):
        txt = ""
        txt += "{}\n".format(len(self.flowList))
        for flow in self.flowList:
            txt += flow.getText()
        return txt
    def saveText(self,out_path):
        with open(out_path,"w") as f:
            f.write(self.getText())
    
    def exportAsJson(self,out_path):
        def write(dic):
            json.dump(dic, open(out_path, 'w'), indent=2)
        write(
            {
                'flows':[flow.exportAsDict() for flow in self.flowList],
            }
        )
        
    @classmethod
    def fromText(cls,lines,roadDS=None):
        flowDSC=FlowDataSetCreater(roadDS)
        numFlow=int(lines[0].rstrip('n'))
        for idx in range(numFlow):
            flow=Flow.fromText(lines[1+3*idx:4+3*idx])
            flowDSC.flowList.append(flow)
        return flowDSC.compile()
    
    @staticmethod
    def createFromFlowFile(path,roadDS=None):
        with open(path, 'r') as f:
            lines = f.readlines()
        return FlowDataSet.fromText(lines,roadDS)
    
    
class FlowDataSetCreater():
    def __init__(self,roadDS=None):
        self.flowList = []
        self.roadDS=roadDS

    def appendByRoadIdList(self, start, end, interval, roadIdlist):
        flow=Flow(start,end,interval,roadIdlist)
        if self.roadDS is not None:
            self.roadDS.assertFlow(flow)
        self.flowList.append(flow)
        
    def appendByInterIdList(self, start, end, interval, interIdlist):
        if self.roadDS is None:
            raise Exception("feed roadDS when constructing FlowDataSetCreater to use appendByInterIdList")
        
        roadList=self.roadDS.solveRoute(interIdlist)
        self.appendByRoadIdList(start, end, interval, [road.roadId for road in roadList])
        
    def compile(self):
        return FlowDataSet(self.flowList)
