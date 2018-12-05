####this file is only used for continuous evaluation test!
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

#### NOTE kpi.py should shared in models in some way!!!!

cgan_d_train_cost_kpi = CostKpi('cgan_d_train_cost', 0.02, 0, actived=True, desc='train cost of discriminator')
cgan_g_train_cost_kpi = CostKpi('cgan_g_train_cost', 0.02, 0, actived=True, desc='train cost of generator')
cgan_train_speed_kpi = DurationKpi(
    'cgan_duration',
    0.05,
    0,
    actived=True,
    unit_repr='second',
    desc='train time used in one GPU card')
dcgan_d_train_cost_kpi = CostKpi('dcgan_d_train_cost', 0.02, 0, actived=True, desc='train cost of discriminator')
dcgan_g_train_cost_kpi = CostKpi('dcgan_g_train_cost', 0.02, 0, actived=True, desc='train cost of generator')
dcgan_train_speed_kpi = DurationKpi(
    'dcgan_duration',
    0.05,
    0,
    actived=True,
    unit_repr='second',
    desc='train time used in one GPU card')


tracking_kpis = [dcgan_d_train_cost_kpi, dcgan_g_train_cost_kpi, 
        dcgan_train_speed_kpi, cgan_d_train_cost_kpi, cgan_g_train_cost_kpi, cgan_train_speed_kpi]


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
        fs = line.strip().split(',')
        print(fs)
        if len(fs) == 3 and fs[0] == 'kpis':
            kpi_name = fs[1]
            kpi_value = float(fs[2])
            print("kpi {}={}".format(kpi_name, kpi_value))
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
#    print("*****")
#    print(log)
#    print("****")
    log_to_ce(log)
