# The notes on the updates of PaddlePaddle baseline

## Updates

We implement a BiDAF model with PaddlePaddle. Note that we have an update on the PaddlePaddle baseline (Feb 25, 2019). In this document, we give the details of the major updates:

### 1 Paragraph Extraction

The first update is that we incorporate a strategy of paragraph extraction to improve the model performance (see the file `paddle/para_extraction.py`). A similar strategy has been used in the Top-1 system (Liu et al. 2018) at [2018 Machine Reading Challenge](http://mrc2018.cipsc.org.cn/). 

The original baseline of DuReader (He et al. 2018) employed a simple strategy to select paragraphs for model training and testing. However, the paragraphs that includes the true answers may not be selected. Hence, we want to incorporate as much information for the answer extraction as possible. 

The detail of the new strategy of paragraph extraction is as follows. We apply the new paragraph extraction strategy on each document. For each document, 
 - We remove the duplicated paragraphs in the document.
 - We concatenate the title and all paragraphs in the document with a pre-defined splitter if it is shorter than a predefined maximum length. Otherwise, 
	- We compute F1 score of each paragraph relative to the question; 
	- We concatenate the title and the top-K paragraphs (by F1 score) with a pre-defined splitter to form an extracted paragraph that should be shorter than the predefined maximum length.

### 2 The Prior of Document Ranking

We also introduce the prior of document ranking from search engine (see line #176 in `paddle/run.py`). The documents in DuReader are collected from the search results. Hence, the prior scores of document ranking is an important feature. We compute the prior scores from the training data and apply the prior scores in the testing stage.  

## Reference

- Liu, J., Wei, W., Sun, M., Chen, H., Du, Y. and Lin, D., 2018. A Multi-answer Multi-task Framework for Real-world Machine Reading Comprehension. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 2109-2118).

- He, W., Liu, K., Liu, J., Lyu, Y., Zhao, S., Xiao, X., Liu, Y., Wang, Y., Wu, H., She, Q. and Liu, X., 2017. Dureader: a chinese machine reading comprehension dataset from real-world applications. arXiv preprint arXiv:1711.05073.

