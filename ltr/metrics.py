import numpy as np
import unittest


def ndcg(score_list):
    """
    measure the ndcg score of order list
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    parameter:
        score_list: np.array, shape=(sample_num,1)

    e.g. predict rank score list :
    >>> scores =  [3, 2, 3, 0, 1, 2] 
    >>> ndcg_score = ndcg(scores)
    """

    def dcg(score_list):
        n = len(score_list)
        cost = .0
        for i in range(n):
            cost += float(np.power(2, score_list[i])) / np.log((i + 1) + 1)
        return cost

    dcg_cost = dcg(score_list)
    score_ranking = sorted(score_list, reverse=True)
    ideal_cost = dcg(score_ranking)
    return dcg_cost / ideal_cost


class TestNDCG(unittest.TestCase):
    def test_array(self):
        a = [3, 2, 3, 0, 1, 2]
        value = ndcg(a)
        self.assertAlmostEqual(0.9583, value, places=3)


if __name__ == '__main__':
    unittest.main()
