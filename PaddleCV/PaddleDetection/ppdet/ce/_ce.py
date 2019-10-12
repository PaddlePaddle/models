# this file is only used for continuous evaluation test!

import os
import sys

sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi

CascadeRCNN_train_loss_kpi = CostKpi('CascadeRCNN_train_loss', 0.08, 0, actived=True, desc='train cost')
CascadeRCNN_train_time_kpi = DurationKpi('CascadeRCNN_train_time', 0.08, 0, actived=True,
                                         desc='train speed in 8 GPU card')
FasterRCNN_train_loss_kpi = CostKpi('FasterRCNN_train_loss', 0.3, 0, actived=True, desc='train cost')
FasterRCNN_train_time_kpi = DurationKpi('FasterRCNN_train_time', 0.08, 0, actived=True,
                                        desc='train speed in 8 GPU card')
MaskRCNN_train_loss_kpi = CostKpi('MaskRCNN_train_loss', 0.08, 0, actived=True, desc='train cost')
MaskRCNN_train_time_kpi = DurationKpi('MaskRCNN_train_time', 0.08, 0, actived=True, desc='train speed in 8 GPU card')
YOLOv3_train_loss_kpi = CostKpi('YOLOv3_train_loss', 0.3, 0, actived=True, desc='train cost')
YOLOv3_train_time_kpi = DurationKpi('YOLOv3_train_time', 0.1, 0, actived=True, desc='train speed in 8 GPU card')
tracking_kpis = [CascadeRCNN_train_loss_kpi, CascadeRCNN_train_time_kpi, FasterRCNN_train_loss_kpi,
                 FasterRCNN_train_time_kpi, MaskRCNN_train_loss_kpi, MaskRCNN_train_time_kpi, YOLOv3_train_loss_kpi,
                 YOLOv3_train_time_kpi]


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
