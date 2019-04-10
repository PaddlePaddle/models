"""
Evaluation
"""

import sys
import six
import numpy as np
from sklearn.metrics import average_precision_score

def evaluate_ubuntu(file_path):
    """
    Evaluate on ubuntu data
    """
    def get_p_at_n_in_m(data, n, m, ind):
        """
        Recall n at m
        """
        pos_score = data[ind][0]
        curr = data[ind:ind + m]
        curr = sorted(curr, key=lambda x: x[0], reverse=True)
    
        if curr[n - 1][0] <= pos_score:
            return 1
        return 0

    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            tokens = line.split("\t")

            if len(tokens) != 2:
                continue

            data.append((float(tokens[0]), int(tokens[1])))

    #assert len(data) % 10 == 0

    p_at_1_in_2 = 0.0
    p_at_1_in_10 = 0.0
    p_at_2_in_10 = 0.0
    p_at_5_in_10 = 0.0

    length = len(data) // 10

    for i in six.moves.xrange(0, length):
        ind = i * 10
        assert data[ind][1] == 1

        p_at_1_in_2 += get_p_at_n_in_m(data, 1, 2, ind)
        p_at_1_in_10 += get_p_at_n_in_m(data, 1, 10, ind)
        p_at_2_in_10 += get_p_at_n_in_m(data, 2, 10, ind)
        p_at_5_in_10 += get_p_at_n_in_m(data, 5, 10, ind)

    result_dict = {
        "1_in_2": p_at_1_in_2 / length,
        "1_in_10": p_at_1_in_10 / length,
        "2_in_10": p_at_2_in_10 / length,
        "5_in_10": p_at_5_in_10 / length}

    return result_dict


def evaluate_douban(file_path):
    """
    Evaluate douban data
    """
    def mean_average_precision(sort_data):
        """
        Evaluate mean average precision
        """
        count_1 = 0
        sum_precision = 0
        for index in six.moves.xrange(len(sort_data)):
            if sort_data[index][1] == 1:
                count_1 += 1
                sum_precision += 1.0 * count_1 / (index + 1)
        return sum_precision / count_1
    
    def mean_reciprocal_rank(sort_data):
        """
        Evaluate MRR
        """
        sort_lable = [s_d[1] for s_d in sort_data]
        assert 1 in sort_lable
        return 1.0 / (1 + sort_lable.index(1))
    
    def precision_at_position_1(sort_data):
        """
        Evaluate precision
        """
        if sort_data[0][1] == 1:
            return 1
        else:
            return 0
    
    def recall_at_position_k_in_10(sort_data, k):
        """"
        Evaluate recall
        """
        sort_lable = [s_d[1] for s_d in sort_data]
        select_lable = sort_lable[:k]
        return 1.0 * select_lable.count(1) / sort_lable.count(1)
    
    def evaluation_one_session(data):
        """
        Evaluate one session
        """
        sort_data = sorted(data, key=lambda x: x[0], reverse=True)
        m_a_p = mean_average_precision(sort_data)
        m_r_r = mean_reciprocal_rank(sort_data)
        p_1 = precision_at_position_1(sort_data)
        r_1 = recall_at_position_k_in_10(sort_data, 1)
        r_2 = recall_at_position_k_in_10(sort_data, 2)
        r_5 = recall_at_position_k_in_10(sort_data, 5)
        return m_a_p, m_r_r, p_1, r_1, r_2, r_5

    sum_m_a_p = 0
    sum_m_r_r = 0
    sum_p_1 = 0
    sum_r_1 = 0
    sum_r_2 = 0
    sum_r_5 = 0
    i = 0
    total_num = 0
    with open(file_path, 'r') as infile:
        for line in infile:
            if i % 10 == 0:
                data = []

            tokens = line.strip().split('\t')
            data.append((float(tokens[0]), int(tokens[1])))
            if i % 10 == 9:
                total_num += 1
                m_a_p, m_r_r, p_1, r_1, r_2, r_5 = evaluation_one_session(data)
                sum_m_a_p += m_a_p
                sum_m_r_r += m_r_r
                sum_p_1 += p_1
                sum_r_1 += r_1
                sum_r_2 += r_2
                sum_r_5 += r_5
            i += 1

    result_dict = {
        "MAP": 1.0 * sum_m_a_p / total_num,
        "MRR": 1.0 * sum_m_r_r / total_num,
        "P_1": 1.0 * sum_p_1 / total_num,
        "1_in_10": 1.0 * sum_r_1 / total_num,
        "2_in_10": 1.0 * sum_r_2 / total_num,
        "5_in_10": 1.0 * sum_r_5 / total_num}
    return result_dict


