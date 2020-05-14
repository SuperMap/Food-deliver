# -*- coding: utf-8 -*-
# @Author  : woleto
# @Time    : 2020/5/11 10:23
import json
import pickle
from functools import reduce
from typing import Dict, List
import numpy as np
import torch

from demo.dto import Courier
from DeliverEnv import DeliverEnv

def getOne():
    with open('data/test.txt', 'r') as r:
        lines = r.read().splitlines()
        print(type(lines))
        while True:
            for line in lines:
                yield line


def getOne1(loop=4):
    for i in range(loop):
        with open('data/test.txt', 'r') as r:
            line = r.readline()
            while line:
                yield line.replace('\n', '')
                line = r.readline()
            yield None

# with open('data/test.txt') as r:
#     lines = r.read().splitlines()
#     for line in lines:
#         json_data = json.loads(line)
#         json_data


if __name__ == '__main__':
    # with open('data/context.pickle', 'rb') as r:
    #     context = pickle.load(r)
    #     print(context.areaId)
    deliverEnv = DeliverEnv('680507')
    quest = deliverEnv.getOneJsonRequest()
    for i in range(100):
        print(next(quest))
