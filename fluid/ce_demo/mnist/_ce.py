# this file is only used for continuous evaluation test!

import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

# NOTE kpi.py should shared in models in some way!!!!

train_cost_kpi = CostKpi('train_cost', 0.02, actived=True)
test_acc_kpi = AccKpi('test_acc', 0.005, actived=True)
train_duration_kpi = DurationKpi('train_duration', 0.06, actived=True)
train_acc_kpi = AccKpi('train_acc', 0.005, actived=True)

tracking_kpis = [
    train_acc_kpi,
    train_cost_kpi,
    test_acc_kpi,
    train_duration_kpi,
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
    for line in log.split():
        fs = line.strip().split('\t')
        if len(fs) == 2:
            kpi_name = fs[0]
            kpi_value = float(fs[1])
            yield kpi_name, kpi_value


def log_to_ce(log):
    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for (kpi_name, kpi_value) in parse_log(log):
        kpi_tracker[kpi_name].add_record(kpi_value)


if __name__ == '__main__':
    log = sys.stdin.read()
    log_to_ce(log) 
    
