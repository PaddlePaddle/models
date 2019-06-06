Proactive Conversation
=============================
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Motivation
Human-machine conversation is one of the most important topics in artificial intelligence (AI) and has received much attention across academia and industry in recent years. Currently dialogue system is still in its infancy, which usually converses passively and utters their words more as a matter of response rather than on their own initiatives, which is different from human-human conversation. We believe that the ability of proactive conversation of machine is the breakthrough of human-like conversation.
# What we do ?
* We set up a new conversation task, named ___Proactive Converstion___, where machine proactively leads the conversation following a given goal. For more details of the task, we refer readers to the paper of [Proactive Human-Machine Conversation with Explicit Conversation Goals]() which was accepted by ACL 2019 
* We also created a new conversation dataset named [DuConv](https://ai.baidu.com/broad/subordinate?dataset=duconv) , and made it publicly available to facilitate the development of proactive conversation systems.
* We established retrival-based and generation-based baseline systems for DuConv, which are available in this repo.
* In addition, we held a [competition](http://lic2019.ccf.org.cn/talk) to encourage more researchers to work in this direction.

3.It aims at testing machines’ ability to conduct human-like conversations.<br>
Please refer to [competition website](http://lic2019.ccf.org.cn/talk) for details of the competition.
# about the task
Given a dialogue goal g and a set of topic-related background knowledge M = f<sub>1</sub> ,f<sub>2</sub> ,..., f<sub>n</sub> , a participating system is expected to output an utterance "u<sub>t</sub>" for the current conversation H = u<sub>1</sub>, u<sub>2</sub>, ..., u<sub>t-1</sub>, which keeps the conversation coherent and informative under the guidance of the given goal. During the dialogue, a participating system is required to proactively lead the conversation from one topic to another. The dialog goal g is given like this: "Start->Topic_A->TOPIC_B", which means the machine should lead the conversation from any start state to topic A and then to topic B. The given background knowledge includes knowledge related to topic A and topic B, and the relations between these two topics.<br>
Please refer to [task description](https://github.com/baidu/knowledge-driven-dialogue/blob/master/task_description.pdf) for details of the task.
# about the baseline
We provide retrieval-based and generation-based baseline systems. Both systems were implemented by [PaddlePaddle](http://paddlepaddle.org/) (the Baidu deeplearning framework) and [Pytorch](https://pytorch.org/) (the Facebook deeplearning framework). The performance of the two systems is as follows:

| baseline system | F1/BLEU1/BLEU2 | DISTINCT1/DISTINCT2 |
| ------------- | ------------ | ------------ |
| retrieval-based | 31.72/0.291/0.156 | 0.118/0.373 |
| generation-based | 32.65/0.300/0.168 | 0.062/0.128 |
