# SS-MARL

## Overview

Implementation of *Scalable Safe Multi-Agent Reinforcement Learning for Multi-Agent System*

## Dependencies & Installation

We recommend to use CONDA to install the requirements:
```shell
conda create -n SSMARL python=3.7
conda activate SSMARL
pip install -r requirements.txt
```
Install SS-MARL:
```shell
pip intall -e.
```

## Run

### Environment
We have modified the Multi-agent Particle Environment (MPE) to better facilitate Safe Multi-Agent Reinforcement Learning (MARL) algorithms. The modifications are implemented in the `ssmarl/envs/mpe_env/multiagent/environment.py` file, 
where we have defined three types of environments to accommodate different MARL algorithms.
* `MultiAgentEnv`:   
Base class for other environment classes.   
Only fixed size observations of the agent are allowed.   
Dose not consider cost constraints.   
Suitable for basic MARL algorithms, such as MAPPO, MADDPG, MATD3, etc.
* `MultiAgentConstrainEnv` :   
Only fixed size observations of the agent are allowed.   
Consider cost constraints.   
Suitable for basic Safe MARL algorithms, such as MACPO, MAPPO-Lagrangian, etc.
* `MultiAgentGraphConstrainEnv` :   
Variable-sized observations setting in SS-MARL are allowed.   
Consider cost constraints.   
Suitable for SS-MARL.

### Scenario
Users can use the cooperative navigation scenario described in the paper or create and use custom scenarios. The scenario files should be placed in the `ssmarl/envs/mpe_env/multiagent/scenarios` directory.

### Parameter
All parameters and their detailed functions are described in the `ssmarl/config.py` file. Users can modify the default values for different training or testing tasks as needed.

### Train
To train the model using SS-MARL, follow these steps:
```shell
cd ssmarl/scripts
python train_mpe.py
```
The training logs will be saved in the `ssmarl/results` directory. 
If you have enabled Weights & Biases (wandb), the logs will also be uploaded to your wandb project.

### Test
To test the model trained by SS-MARL, follow these steps:
```shell
cd ssmarl/scripts
python render_mpe.py
```
Setting `use_render` to `True` will enable rendering of the environment in a separate window. Additionally, if `save_gifs` is set to `True`, the generated gifs will be saved to the `ssmarl/results` directory.

### Model

To fine-tune or test a model, users can save their model files in the `ssmarl/model` directory, where they will be automatically restored. Additionally, pre-trained SS-MARL(PS) model files are already available in the same directory for immediate use.

## Demo 1
This demo shows the strong scalability of SS-MARL. Despite being trained with only 3 agents, the models are capable of more complex, randomly generated scenarios, even scaling up to a challenge involving 96 agents.  
### Train on cooperative navigation with 3 agents

![3 agents](https://github.com/Qeneb/SS-MARL/demo/navigation/3agents.gif "3 agents")

### Zero-shot transfer to cooperative navigation with 6, 12, 24, 48, 96 agents

<img src="https://github.com/Qeneb/SS-MARL/demo/navigation/6agents.gif" alt="6agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/navigation/12agents.gif" alt="12agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/navigation/24agents.gif" alt="24agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/navigation/48agents.gif" alt="48agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/navigation/96agents.gif" alt="96agents" style="width: 250px; height: 250px;" />  

### Zero-shot transfer to cooperative formation with 3, 6, 12 agents

<img src="https://github.com/Qeneb/SS-MARL/demo/formation/3agents.gif" alt="3agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/formation/6agents.gif" alt="6agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/formation/12agents.gif" alt="12agents" style="width: 250px; height: 250px;" />

### Zero-shot transfer to cooperative line with 3, 6, 12 agents

<img src="https://github.com/Qeneb/SS-MARL/demo/line/3agents.gif" alt="3agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/line/6agents.gif" alt="6agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/line/12agents.gif" alt="12agents" style="width: 250px; height: 250px;" />

## Demo 2
This demo shows that SS-MARL can also handle other cooperative tasks. We retrain SS-MARL on cooperative formation and cooperative line tasks. Here is the introduction of two cooperative tasks.
* In the Formation task, a scenario is set where $N$ agents are required to arrange themselves around a single landmark, which is positioned at the center of an $N$-sided regular polygon. The agents receive rewards at each time step based on their proximity to their designated positions. Additionally, they incur costs for colliding with one another. These positions are determined by solving a linear assignment problem at each time step, which takes into account the number of agents present in the environment and the desired radius of the polygon. For our experiments, we have set the target radius to a value of 0.5.
* In the Line task, a setup is presented involving $N$ agents and a pair of landmarks. The objective for the agents is to disperse themselves evenly along a straight line that stretches between the two landmarks. As with the Formation environment, the agents earn rewards based on their nearness to the positions they are supposed to occupy. These positions are determined through the resolution of a linear sum assignment problem at every time step.

### Train on cooperative formation with 3, 4, 5, 6 agents

<img src="https://github.com/Qeneb/SS-MARL/demo/formation/retrain_3agents.gif" alt="3agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/formation/retrain_4agents.gif" alt="4agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/formation/retrain_5agents.gif" alt="5agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/formation/retrain_6agents.gif" alt="6agents" style="width: 250px; height: 250px;" />

### Train on cooperative line with 3, 4, 5, 6 agents

<img src="https://github.com/Qeneb/SS-MARL/demo/line/retrain_3agents.gif" alt="3agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/line/retrain_4agents.gif" alt="4agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/line/retrain_5agents.gif" alt="5agents" style="width: 250px; height: 250px;" />  <img src="https://github.com/Qeneb/SS-MARL/demo/line/retrain_6agents.gif" alt="6agents" style="width: 250px; height: 250px;" />


