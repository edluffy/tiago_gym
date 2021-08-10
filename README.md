# tiago_gym
A ROS package to run Reinforcement Learning experiments, particularly pick and place tasks, on the TIAGo robot. Uses Gazebo, Rviz and MoveIt! (for motion planning)

![rviz-showcase](https://user-images.githubusercontent.com/28115337/128855778-1333fb2a-a6ac-47d0-8d59-ccc5798a2c32.gif)

## Environments

| TiagoSimpleEnv-v0        | TiagoReachEnv-v0         |
| ------------------------ | ------------------------ |
![tiago_gym_simple](https://user-images.githubusercontent.com/28115337/128849740-a8ac397e-a904-4f41-b9e0-b3ad4ac71f57.gif) |  ![tiago_gym_reach](https://user-images.githubusercontent.com/28115337/128849822-865f9c43-dc75-4863-a84f-0b6a754dd04e.gif)|

### TiagoSimpleEnv-v0
This is a simple test environment in which the robot gripper must move to a discrete goal position in 3D space (essentially a 3D gridworld). Example usage:

``` python
def train_simple_env():
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
def train_reach_env():
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
