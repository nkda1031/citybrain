# how to import or load local files
import os
import sys
cwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(cwd+"/util")
from simulation import BridgeAgent
from strategy import Strategy,StrategyActionSolver

###parameters for DNN agent
dnnSavedModelDirName="round2_18_18_5_r1"
dnnNumSegmentInbound=18
dnnNumSegmentOutbound=18
##########################
savedModelPath=cwd+"/"+dnnSavedModelDirName
print("************************************")
print("load saved model:",savedModelPath)
print("************************************")

strategy=Strategy.createStrategy(
    'dnn_saved_run_distance',{
        'savedModelPath':savedModelPath,
        'numSegmentInbound':dnnNumSegmentInbound,
        'numSegmentOutbound':dnnNumSegmentOutbound,
    }
)


scenario_dirs = [
    "test"
]
agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    agent_specs[k] = BridgeAgent(
        StrategyActionSolver(strategy)
    )