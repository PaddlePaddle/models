# Reproduce DQN, DoubleDQN, DuelingDQN model with fluid version of PaddlePaddle

+ DQN in:
[Human-level Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
+ DoubleDQN in:
[Deep Reinforcement Learning with Double Q-Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389)
+ DuelingDQN in:
[Dueling Network Architectures for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/wangf16.html)

# Atari benchmark & performance
## [Atari games introduction](https://gym.openai.com/envs/#atari)

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
    python train.py --rom ./rom_files/pong.bin --use_cuda

    # To train a model for Pong with DoubleDQN
    python train.py --rom ./rom_files/pong.bin --use_cuda --alg DoubleDQN

    # To train a model for Pong with DuelingDQN
    python train.py --rom ./rom_files/pong.bin --use_cuda --alg DuelingDQN
    ```

To train more games, can install more rom files from [here](https://github.com/openai/atari-py/tree/master/atari_py/atari_roms)

+ Start Testing:
    ```
    # Play the game with saved model and calculate the average rewards
    python play.py --rom ./rom_files/pong.bin --use_cuda --model_path ./saved_model/DQN-pong/stepXXXXX

    # Play the game with visualization
    python play.py --rom ./rom_files/pong.bin --use_cuda --model_path ./saved_model/DQN-pong/stepXXXXX --viz 0.01
    ```
