## Introduction
Sequence is an input data type faced by many machine learning and data mining tasks. Taking Natural Language Processing task as an example, sentence is composed of words, and paragraph is composed of sentences. As a result, a paragraph can be seen as a nested sequence (or called: double sequence), and each element of the sequence is a sequence.

Double sequence is a very flexible data organization method supported by PaddlePaddle, which can help us better describe more complex data such as paragraphs, multiple rounds of dialogues. With a double-layer sequence as input, we can design a hierarchical network to better accomplish some complex tasks.

This unit will introduce how to use a double sequence in PaddlePaddle.

- [Text Classification Based on Double Sequence](https://github.com/PaddlePaddle/models/tree/develop/nested_sequence/text_classification)
