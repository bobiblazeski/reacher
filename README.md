# Reacher
Navigation project from Udacity Deep Reinforcement Learning Nanodegree.
It demonstrates how to teach an agent to collect yellow bananas while avoiding blue bananas. 

## Installation
### Install deep reinforcement learning repository
1. Clone [deep reinforcement learning repository](https://github.com/udacity/deep-reinforcement-learning)
2. Fallow the instructions to install necessary [dependencies](https://github.com/udacity/deep-reinforcement-learning#dependencies)
### Download the Unity Environment
1. Download environment for your system into this repository root

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)

* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)

* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)

* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

* Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip)

2. Unzip (or decompress) the archive
### Run the project
1. Start the jupyter server
2. Open the Continuous_Control.ipynb notebook
3. Change the kernel to drlnd
4. You should be able to run all the cells

## Environment
This project uses the Unity based environment prepared by the Udacity  team.

There are 20 agents interacting with the environment.

The actions space is of continous with shape of 4 each between [-1,+1]:

The state is represented as a vector of 33 dimensions.

The environment gives a rewards of  between [0, 1] for reacher that correctly positios the hand
regarding the circling ball.

## Weights
The directory `saves` contains saved weights for 2 different agents:

* `96_96_108_actor.pth` & `96_96_108_critic.pth` - Agent that learned from scratch in 108 episodes
* `96_96_80_actor.pth` & `96_96_80_critic.pth` - Agent that learned from above agent experience in 80 episodes
* `48_48_actor_71.pth` & `48_48_critic_71.pth` - Smaller agent that learned from above agent experience 
* `96_96_2491_actor.pth` & `96_96_2491_critic.pth` - Learned from scratch from single agent version

Naming convention `Fully connected layer 1`_`Fully connected layer 2`_`Episodes`_`[actor|critic]`.pth

## Credits
Most of the code is based on the Udacity code for DDPG. I've adapted some of the code by   [akhiadber](https://github.com/akhiadber/DeepRL_Continuous_Control), which adds batch normalization & training function.

