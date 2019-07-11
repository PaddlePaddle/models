### This file is only used for continuous evaluation test!
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi

train_cost_1card_kpi = CostKpi(
    'train_cost_1card', 0.02, 0, actived=True, desc='train cost')
train_duration_1card_kpi = DurationKpi(
    'train_duration_1card', 0.1, 0, actived=True, desc='train duration')
train_cost_8card_kpi = CostKpi(
    'train_cost_8card', 0.02, 0, actived=True, desc='train cost')
train_duration_8card_kpi = DurationKpi(
    'train_duration_8card', 0.1, 0, actived=True, desc='train duration')
tracking_kpis = [
    train_cost_1card_kpi, train_duration_1card_kpi, train_cost_8card_kpi,
    train_duration_8card_kpi
]


def parse_log(log):
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
    log_to_ce(log)
