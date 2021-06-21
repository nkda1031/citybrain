# how to import or load local files
import os
import sys
cwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(cwd+"/util")
from simulation import BridgeAgent
from strategy import Strategy,StrategyActionSolver,RandomMixStrategy

###parameters for DNN agent
dnnSavedModelDirName="round2_18_18_5_r1"
dnnNumSegmentInbound=18
dnnNumSegmentOutbound=18
###parameters for rule base agent
timeThresForCalcReward=10.2
prohibitDecreasingGoalDistance=True
prohibitDecreasingSpeed=True
###parameters for mixing strategy 
dnnStrategyRatio=0.6
ruleStrategyRatio=0.1
##########################
savedModelPath=cwd+"/"+dnnSavedModelDirName
print("************************************")
print("load saved model:",savedModelPath)
print("************************************")

dnnStrategy=Strategy.createStrategy(
    'dnn_saved_run_distance',{
        'savedModelPath':savedModelPath,
        'numSegmentInbound':dnnNumSegmentInbound,
        'numSegmentOutbound':dnnNumSegmentOutbound,
    }
)
ruleStrategy=Strategy.createStrategy(
    'run_distance',{
        'timeThresForCalcReward':timeThresForCalcReward,
        'prohibitDecreasingGoalDistance':prohibitDecreasingGoalDistance,
        'prohibitDecreasingSpeed':prohibitDecreasingSpeed,
    }
)

strategy=RandomMixStrategy({
    dnnStrategy:dnnStrategyRatio,
    ruleStrategy:ruleStrategyRatio,
})

scenario_dirs = [
    "test"
]
agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    agent_specs[k] = BridgeAgent(
        StrategyActionSolver(strategy)
    )