Data
=====

This dataset is for our paper: ARNOR: Attention Regularization based Noise Reduction for Distant Supervision Relation Classification. This test set is for sentence-level evaluation.

The original data is from the dataset in the paper: Cotype: Joint extraction of typed entities and relations with knowledge bases. It is a distant supervision dataset from NYT (New York Time). And the test set is annotated by humans. However the number of positive instances in test set is small. We revise and annotate more test data based on it.

In a data file, each line is a json string. The content is like

    {
        "sentText": "The source sentence text",
        "relationMentions": [
                                {
                                    "em1Text": "The first entity in relation",
                                    "em2Text": "The second entity in relation",
                                    "label": "Relation label",
                                    "is_noise": false # only occur in test set
                                },
                                ...
                            ],
        "entityMentions":     [
                                {
                                    "text": "Entity words",
                                    "label": "Entity type",
                                    ...
                                },
                                ...
                            ]
        ...
    }

Data version 1.0.0
=====

This version of dataset is the original one applied in our paper, which includes four files: train.json, test.json, dev_part.json, and test_part.json. Here dev_part.json and test_part.json are from test.json. This dataset can be downloaded here: https://baidu-nlp.bj.bcebos.com/arnor_dataset-1.0.0.tar.gz


Data version 2.0.0
=====

More test date are coming soon ......
