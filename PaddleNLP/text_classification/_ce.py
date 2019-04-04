####this file is only used for continuous evaluation test!

import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

#### NOTE kpi.py should shared in models in some way!!!!

train_acc_kpi = AccKpi('train_acc', 0.005, actived=True)
train_cost_kpi = CostKpi('train_cost', 0.005, actived=False)
train_duration_kpi = DurationKpi('train_duration', 0.05, actived=True)

tracking_kpis = [
    train_acc_kpi,
    train_cost_kpi,
    train_duration_kpi,
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
    print("*****")
    print(log)
    print("****")
    log_to_ce(log)
