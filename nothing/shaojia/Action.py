# -*- coding: utf-8 -*-
# @Author  : woleto
# @Time    : 2020/5/9 14:47
from typing import List

from nothing.shaojia.dto import Courier, CourierPlan, Order

class Action(object):
    def __init__(self):
        pass

    def chooseAction(self, couriers: List[Courier], courierPlans: List[CourierPlan], orders: List[Order]):
        """
        目标：已有骑手已完成的动作，依次加入可以加入的节点，形成若干图，放进图网络，得到合理值。
        :param orders: 所有订单信息
        :param couriers: 所有骑手的信息
        :param courierPlans: 所有骑手已完成的动作
        :return:
        """

        pass

