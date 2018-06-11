# Reproduce DQN, DoubleDQN, DuelingDQN model with fluid version of PaddlePaddle

+ DQN in:
[Human-level Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
+ DoubleDQN in:
[Deep Reinforcement Learning with Double Q-Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389)
+ DuelingDQN in:
[Dueling Network Architectures for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/wangf16.html)

# Atari benchmark & performance
+ [Atari 2600 games](https://gym.openai.com/envs/#atari)

+ Pong game result
![DQN result](assets/dqn.png)

# How to use
+ Dependencies:
    + python2.7
    + gym
    + tqdm
    + paddlepaddle-gpu==0.12.0

+ Start Training:
    ```
    # To train a model for Pong game with gpu (use DQN model as default)
    python train.py --rom pong.bin --use_cuda
 
    # To train a model for Pong with DoubleDQN
    python train.py --rom pong.bin --use_cuda --rl DoubleDQN
 
    # To train a model for Pong with DuelingDQN
    python train.py --rom pong.bin --use_cuda --rl DuelingDQN
    ```

+ Start Testing:
    ```
    # Play the game with saved model and calculate the average rewards
    python play.py --rom pong.bin --use_cuda --model_path ./saved_model/DQN-pong/step600872
    ```
