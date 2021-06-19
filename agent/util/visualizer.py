import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

def displayRoadNetStatistics(roadNet):
    plt.tight_layout()

    roadSeg_df=pd.DataFrame(roadNet.roadDataSet.getRoadSegmentStat())
    print("histgram of RoadSegment")
    roadSeg_df.hist(figsize=(15, 3),layout=(1,3),bins=50)
    plt.show()
    
    print("RoadSegment stat")
    display(roadSeg_df["speedLimit"].value_counts())

    road_df=pd.DataFrame(roadNet.roadDataSet.getRoadStat())
    print("Road stat")
    display(road_df["permissibleMovementList"].value_counts())

    inter_df=pd.DataFrame(roadNet.roadDataSet.getIntersectionStat())
    print("Intersection stat")
    display(inter_df["numRoadSegment"].value_counts())

    signal_df=pd.DataFrame(roadNet.intersectionDataSet.getSignalStat())
    print("Signal stat")
    display(signal_df["numRoadSegment"].value_counts())