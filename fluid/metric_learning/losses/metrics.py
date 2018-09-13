import numpy as np
def recall_topk(fea, lab, k = 1):
    fea = np.array(fea)
    fea = fea.reshape(fea.shape[0], -1)
    n = np.sqrt(np.sum(fea**2, 1)).reshape(-1, 1)
    fea = fea/n
    a = np.sum(fea ** 2, 1).reshape(-1, 1)
    b = a.T
    ab = np.dot(fea, fea.T)
    d = a + b - 2*ab
    d = d + np.eye(len(fea)) * 1e8
    sorted_index = np.argsort(d, 1)
    res = 0
    for i in range(len(fea)):
        pred = lab[sorted_index[i][0]]
        if lab[i] == pred:
            res += 1.0
    res = res/len(fea)
    return res
