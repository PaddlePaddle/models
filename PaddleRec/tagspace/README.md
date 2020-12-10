# TagSpace

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。


TagSpace模型的介绍可以参阅论文[#TagSpace: Semantic Embeddings from Hashtags](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/)。

Tagspace模型学习文本及标签的embedding表示，应用于工业级的标签推荐，具体应用场景有feed新闻标签推荐。
