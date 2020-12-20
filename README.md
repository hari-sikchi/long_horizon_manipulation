# Decomposing long horizon problems using learned models

## Installation
```
# Install Arm manipulation environment
cd maddux_gym   
pip install -e .
   
# Install PointNavigation environment   
cd point_nav   
pip install -e .   
```

## Train Reachability Model
```
python tdm/run_agent.py --env=<Maddux-v0/PointNavEnv-v0>
```

## Run Subgoal Planner
```
# For point_nav   
python tdm/subgoal_planner.py     
    
# For Maddux   
python tdm/subgoal_planner_maddux.py   

```


## Run RRT between subgoals and generate statistics