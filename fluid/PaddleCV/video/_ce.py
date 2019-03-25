# this file is only used for continuous evaluation test!

import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi
from kpi import AccKpi


AttentionCluster_youtube8m_each_pass_duration_card1_kpi = DurationKpi('AttentionCluster_youtube8m_each_pass_duration_card1', 0.08, actived=True)
train_AttentionCluster_youtube8m_loss_card1_kpi = CostKpi('train_AttentionCluster_youtube8m_loss_card1', 0.08, actived=False)
train_AttentionCluster_youtube8m_hit_at_one_card1_kpi = CostKpi('train_AttentionCluster_youtube8m_hit_at_one_card1', 0.08, actived=False)
train_AttentionCluster_youtube8m_gap_card1_kpi = CostKpi('train_AttentionCluster_youtube8m_gap_card1', 0.08, actived=False)
train_AttentionCluster_youtube8m_perr_card1_kpi = AccKpi('train_AttentionCluster_youtube8m_perr_card1', 0.08, actived=False)
AttentionCluster_youtube8m_each_pass_duration_card4_kpi = DurationKpi('AttentionCluster_youtube8m_each_pass_duration_card4', 0.08, actived=True)
train_AttentionCluster_youtube8m_loss_card4_kpi = CostKpi('train_AttentionCluster_youtube8m_loss_card4', 0.08, actived=False)
train_AttentionCluster_youtube8m_hit_at_one_card4_kpi = CostKpi('train_AttentionCluster_youtube8m_hit_at_one_card4', 0.08, actived=False)
train_AttentionCluster_youtube8m_gap_card4_kpi = CostKpi('train_AttentionCluster_youtube8m_gap_card4', 0.08, actived=False)
train_AttentionCluster_youtube8m_perr_card4_kpi = AccKpi('train_AttentionCluster_youtube8m_perr_card4', 0.08, actived=False)
TSN_kinetics400_each_pass_duration_card1_kpi = DurationKpi('TSN_kinetics400_each_pass_duration_card1', 0.08, actived=True)
train_TSN_kinetics400_acc1_card1_kpi = AccKpi('train_TSN_kinetics400_acc1_card1', 0.08, actived=False)
train_TSN_kinetics400_acc5_card1_kpi = AccKpi('train_TSN_kinetics400_acc5_card1', 0.08, actived=False)
train_TSN_kinetics400_loss_card1_kpi = CostKpi('train_TSN_kinetics400_loss_card1', 0.08, actived=False)
TSN_kinetics400_each_pass_duration_card4_kpi = DurationKpi('TSN_kinetics400_each_pass_duration_card4', 0.08, actived=True)
train_TSN_kinetics400_acc1_card4_kpi = AccKpi('train_TSN_kinetics400_acc1_card4', 0.08, actived=False)
train_TSN_kinetics400_acc5_card4_kpi = AccKpi('train_TSN_kinetics400_acc5_card4', 0.08, actived=False)
train_TSN_kinetics400_loss_card4_kpi = CostKpi('train_TSN_kinetics400_loss_card4', 0.08, actived=False)


tracking_kpis = [
        AttentionCluster_youtube8m_each_pass_duration_card1_kpi,
        train_AttentionCluster_youtube8m_loss_card1_kpi,
        train_AttentionCluster_youtube8m_hit_at_one_card1_kpi,
        train_AttentionCluster_youtube8m_gap_card1_kpi,
        train_AttentionCluster_youtube8m_perr_card1_kpi,
        AttentionCluster_youtube8m_each_pass_duration_card4_kpi,
        train_AttentionCluster_youtube8m_loss_card4_kpi,
        train_AttentionCluster_youtube8m_hit_at_one_card4_kpi,
        train_AttentionCluster_youtube8m_gap_card4_kpi,
        train_AttentionCluster_youtube8m_perr_card4_kpi,
        TSN_kinetics400_each_pass_duration_card1_kpi,
        train_TSN_kinetics400_acc1_card1_kpi,
        train_TSN_kinetics400_acc5_card1_kpi,
        train_TSN_kinetics400_loss_card1_kpi,
        TSN_kinetics400_each_pass_duration_card4_kpi,
        train_TSN_kinetics400_acc1_card4_kpi,
        train_TSN_kinetics400_acc5_card4_kpi,
        train_TSN_kinetics400_loss_card4_kpi,
        ]


def parse_log(log):
    '''
    This method should be implemented by model developers.

    The suggestion:

    each line in the log should be key, value, for example:

    "
    train_cost\t1.0
    test_cost\t1.0
    train_cost\t1.0
    train_cost\t1.0
    train_acc\t1.2
    "
    '''
    for line in log.split('\n'):
        fs = line.strip().split('\t')
        print(fs)
        if len(fs) == 3 and fs[0] == 'kpis':
            kpi_name = fs[1]
            kpi_value = float(fs[2])
            yield kpi_name, kpi_value


def log_to_ce(log):
    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for (kpi_name, kpi_value) in parse_log(log):
        print(kpi_name, kpi_value)
        kpi_tracker[kpi_name].add_record(kpi_value)
        kpi_tracker[kpi_name].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    log_to_ce(log)
