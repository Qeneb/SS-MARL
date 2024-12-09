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
Users can use the collaborative navigation (CN) scenario described in the paper or create and use custom scenarios. The scenario files should be placed in the `ssmarl/envs/mpe_env/multiagent/scenarios` directory.

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

## Demo
This demo shows the strong scalability of SS-MARL. Despite being trained with only 3 agents, the models are capable of more complex, randomly generated scenarios, even scaling up to a challenge involving 96 agents.  

![3 agents](C:\Users\txzjh\Desktop\SSMARL\demo\3agents.gif)  

![6 agents](C:\Users\txzjh\Desktop\SSMARL\demo\6agents.gif)  


