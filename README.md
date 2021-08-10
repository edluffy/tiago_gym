# 🦾 tiago_gym
A ROS package to run Reinforcement Learning experiments, particularly pick and place tasks, on the TIAGo robot. Uses Gazebo, Rviz and MoveIt! (for motion planning)

<p align="center">
  <img src="https://user-images.githubusercontent.com/28115337/128855778-1333fb2a-a6ac-47d0-8d59-ccc5798a2c32.gif" alt="rviz-showcase" />
</p>

## Installation
Tested on Ubuntu 18.04 only. Beware! Instructions assume familiarity with the ROS [packages](http://wiki.ros.org/Packages) system.
- Install ROS Melodic + TIAGo
    -  http://wiki.ros.org/Robots/TIAGo/Tutorials/Installation/InstallUbuntuAndROS
- Install openai_ros package into your TIAGo workspace
    ``` bash
     cd /home/user/tiago_public_ws/src
     git clone https://bitbucket.org/theconstructcore/openai_ros.git
     cd openai_ros;git checkout version2
     cd /home/user/tiago_public_ws;catkin_make;source devel/setup.bash
    ``` 
- Install tiago_gym package into your TIAGo workspace
    ``` bash
     cd /home/user/tiago_public_ws/src
     git clone https://github.com/edluffy/tiago_gym
     cd /home/user/tiago_public_ws;catkin_make;source devel/setup.bash
    ``` 
- Launch an environment!
    - `roslaunch tiago_gym start_gym.launch`
    - Gazebo and Rviz should launch similarly to gif above.
    - To speed up simulation you can run the command `gz physics –u 0 -s 0.0025` in a separate terminal.
- Running custom training scripts
    - Launch with `roslaunch tiago_gym start_gym.launch train:=false`
    - Run in a separate terminal `python my_training_script.py`. Example snippets are shown in the [Environments](#environments)
section.

## Environments

| TiagoSimpleEnv-v0        | TiagoReachEnv-v0         |
| ------------------------ | ------------------------ |
![tiago_gym_simple](https://user-images.githubusercontent.com/28115337/128849740-a8ac397e-a904-4f41-b9e0-b3ad4ac71f57.gif) |  ![tiago_gym_reach](https://user-images.githubusercontent.com/28115337/128849822-865f9c43-dc75-4863-a84f-0b6a754dd04e.gif)|

### TiagoSimpleEnv-v0
This is a simple test environment in which the robot gripper must move to a discrete goal position in 3D space (essentially a 3D gridworld). Example usage:

``` python
import tiago_simple_env
from agents import dqn

env = gym.make('TiagoSimpleEnv-v0')
o_dims = len(env.observation_space.sample())
a_dims = env.action_space.n

agent = dqn.DQN(env, input_size=o_dims,
        output_size=o_dims, alpha=0.01, epsilon_decay=0.95)
agent.run(100)
```
<table>
<tr><th>Observations </th><th>Actions</th><th>Rewards(Dense!)</th></tr>
<tr><td>
  
|     |                        |
| --- | ---------------------- |
| 0   | x-pos of gripper       |
| 1   | y-pos of gripper       |
| 2   | z-pos of gripper       |
</td><td>

|     |                        |
| --- | ---------------------- |
| 0   | x-pos of gripper + 0.1 |
| 1   | x-pos of gripper - 0.1 |
| 2   | y-pos of gripper + 0.1 |
| 3   | y-pos of gripper - 0.1 |
| 4   | z-pos of gripper + 0.1 |
| 5   | z-pos of gripper - 0.1 |
</td><td>
  
|                  |                   |
| -----------------|------------------ |
| Goal within 0.05 | 10                |
| Else             | -Distance to goal |
</td></tr> </table>


### TiagoReachEnv-v0
A continuous action environment – robot can move a vector distance in any direction to get to the goal. Example usage:

``` python
import tiago_reach_env
from agents import ddpg

env = gym.make('TiagoReachEnv-v0')
o_dims = len(env.observation_space.sample())
a_dims = env.action_space.shape[0]
a_high = env.action_space.high
    
agent = ddpg.DDPG(env, o_dims, a_dims, a_high)
agent.run(100)
 ```

<table>
<tr><th>Observations </th><th>Actions</th><th>Rewards(Dense!)</th></tr>
<tr><td>
  
|     |                                         |
| --- | --------------------------------------- |
| 0   | absolute pos of gripper                 |
| 1   | relative pos  of gripper                |
</td><td>


|     |                        |
| --- | ---------------------- |
| 0   | x-pos of gripper       |
| 1   | x-pos of gripper       |
| 2   | y-pos of gripper       |
</td><td>


|                  |                   |
| -----------------|------------------ |
| Goal within 0.05 | 10                |
| Else             | -Distance to goal |
</td></tr> </table>

## Agents
- Tensorflow implementations of DQN and DDPG can be found in `scripts/agents`.
