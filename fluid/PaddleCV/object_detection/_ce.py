####this file is only used for continuous evaluation test!

import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

#### NOTE kpi.py should shared in models in some way!!!!

train_cost_kpi = CostKpi('train_cost', 0.02, 0, actived=True)
test_acc_kpi = AccKpi('test_acc', 0.01, 0, actived=False)
train_speed_kpi = DurationKpi('train_speed', 0.1, 0, actived=True, unit_repr="s/epoch")
train_cost_card4_kpi = CostKpi('train_cost_card4', 0.02, 0, actived=True)
test_acc_card4_kpi = AccKpi('test_acc_card4', 0.01, 0, actived=False)
train_speed_card4_kpi = DurationKpi('train_speed_card4', 0.1, 0, actived=True, unit_repr="s/epoch")

tracking_kpis = [
    train_cost_kpi,
    test_acc_kpi,
    train_speed_kpi,
    train_cost_card4_kpi,
    test_acc_card4_kpi,
    train_speed_card4_kpi,
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
    #kpi_map = {}
    for line in log.split('\n'):
        fs = line.strip().split('\t')
        print(fs)
        if len(fs) == 3 and fs[0] == 'kpis':
            print("-----%s" % fs)
            kpi_name = fs[1]
            kpi_value = float(fs[2])
            #kpi_map[kpi_name] = kpi_value
            yield kpi_name, kpi_value
    #return kpi_map


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
