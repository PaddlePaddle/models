#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import re
sys.path.append(os.environ['ceroot'])
from kpi import AccKpi

test_acc_top1_kpi = AccKpi(
    'test_acc_top1', 0.02, 0, actived=True, desc='TOP1 ACC')
test_acc_top5_kpi = AccKpi(
    'test_acc_top5', 0.02, 0, actived=True, desc='TOP5 ACC')
tracking_kpis = [test_acc_top1_kpi, test_acc_top5_kpi] 


def parse_log(log):
    '''
    parse log
    '''
    pattern = r"^.*Final eval result: \['acc_top1', 'acc_top5'\]=\[(?P<test_acc_top1>0\.\d+)\s+(?P<test_acc_top5>0\.\d+)\s*\]"
    prog = re.compile(pattern)
    for line in log.split('\n'):
        result = prog.match(line)
        if not result:
            continue
        for kpi_name, kpi_value in result.groupdict().iteritems():
            yield kpi_name, float(kpi_value)


def log_to_ce(log):
    """
    log to ce
    """
    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for(kpi_name, kpi_value) in parse_log(log):
        kpi_tracker[kpi_name].add_record(kpi_value)
        kpi_tracker[kpi_name].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    log_to_ce(log)


