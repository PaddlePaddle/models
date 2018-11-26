####this file is only used for continuous evaluation test!
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

#### NOTE kpi.py should shared in models in some way!!!!

train_acc_top1_kpi = AccKpi(
    'train_acc_top1', 0.02, 0, actived=True, desc='TOP1 ACC')
train_acc_top5_kpi = AccKpi(
    'train_acc_top5', 0.02, 0, actived=True, desc='TOP5 ACC')
train_cost_kpi = CostKpi('train_cost', 0.02, 0, actived=True, desc='train cost')
test_acc_top1_kpi = AccKpi(
    'test_acc_top1', 0.02, 0, actived=True, desc='TOP1 ACC')
test_acc_top5_kpi = AccKpi(
    'test_acc_top5', 0.02, 0, actived=True, desc='TOP5 ACC')
test_cost_kpi = CostKpi('test_cost', 0.02, 0, actived=True, desc='train cost')
train_speed_kpi = DurationKpi(
    'train_speed',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train speed in one GPU card')
train_acc_top1_card4_kpi = AccKpi(
    'train_acc_top1_card4', 0.02, 0, actived=True, desc='TOP1 ACC')
train_acc_top5_card4_kpi = AccKpi(
    'train_acc_top5_card4', 0.02, 0, actived=True, desc='TOP5 ACC')
train_cost_card4_kpi = CostKpi(
    'train_cost_card4', 0.02, 0, actived=True, desc='train cost')
test_acc_top1_card4_kpi = AccKpi(
    'test_acc_top1_card4', 0.02, 0, actived=True, desc='TOP1 ACC')
test_acc_top5_card4_kpi = AccKpi(
    'test_acc_top5_card4', 0.02, 0, actived=True, desc='TOP5 ACC')
test_cost_card4_kpi = CostKpi(
    'test_cost_card4', 0.02, 0, actived=True, desc='train cost')
train_speed_card4_kpi = DurationKpi(
    'train_speed_card4',
    0.05,
    0,
    actived=True,
    unit_repr='seconds/image',
    desc='train speed in four GPU card')
tracking_kpis = [
    train_acc_top1_kpi, train_acc_top5_kpi, train_cost_kpi, test_acc_top1_kpi,
    test_acc_top5_kpi, test_cost_kpi, train_speed_kpi, train_acc_top1_card4_kpi,
    train_acc_top5_card4_kpi, train_cost_card4_kpi, test_acc_top1_card4_kpi,
    test_acc_top5_card4_kpi, test_cost_card4_kpi, train_speed_card4_kpi
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
            print("-----%s" % fs)
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
    print("*****")
    print(log)
    print("****")
    log_to_ce(log)
