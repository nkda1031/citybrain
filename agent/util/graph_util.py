from math import acos,sin,cos,radians

class Node():
    def __init__(self,nodeId,x,y):
        self.nodeId=nodeId
        self.x=x
        self.y=y
        self.linkDict={}
        
    @staticmethod
    def calcDistance(node1,node2,onEarth=False):
        if onEarth: # regard x as longtitude, y as latitude 
            R=6378137 #eearth radious [meter]
            y1=radians(node1.y)
            y2=radians(node2.y)
            x1=radians(node1.x)
            x2=radians(node2.x)
            rad=acos(sin(y1)*sin(y2)+cos(y1)*cos(y2)*cos(x1-x2))
            return round(R*rad)
        else:
            return ((node1.x-node2.x)**2+(node1.y-node2.y)**2)**0.5
    
    def checkLink(self,key):
        if key in self.linkDict:
            if self.linkDict[key].node1==self:
                return 1
            elif self.linkDict[key].node2==self:
                return 2
            else:
                raise Exception("not expected")
        else:
            return 0
class Link():
    def __init__(self,linkId,node1,node2,speedLimit=None):
        self.linkId=linkId
        self.node1=node1
        self.node2=node2
        self.speedLimit=speedLimit

class Graph():
    def __init__(self):
        self.nodeList=[]
        self.linkList=[]
    def appendNode(self,x,y):
        nodeId=len(self.nodeList)
        node=Node(nodeId,x,y)
        self.nodeList.append(node)
        return nodeId
    def appendLink(self,nodeId1,nodeId2,speedLimit=None):
        node1=self.nodeList[nodeId1]
        node2=self.nodeList[nodeId2]
        
        if node1.x==node2.x and node1.y==node2.y:
            raise Exception("x1==x2 and y1==y2")
            
        d=None
        if node1.x==node2.x:
            if node1.y>node2.y:
                d=('S','N')
            else:
                d=('N','S')
        elif node1.y==node2.y:
            if node1.x>node2.x:
                d=('W','E')
            else:
                d=('E','W')
        else:
            raise Exception("x1!=x2 and y1!=y2")
        
        if d[0] in node1.linkDict or d[1] in node2.linkDict:
            raise Exception("already linked")            

        linkId=len(self.linkList)
        link=Link(linkId,node1,node2,speedLimit=speedLimit)
        self.linkList.append(link)
        node1.linkDict[d[0]]=link
        node2.linkDict[d[1]]=link
            
        return linkId
