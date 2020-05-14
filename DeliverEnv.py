# -*- coding: utf-8 -*-
# @Author  : woleto
# @Time    : 2020/5/9 9:34
import json
from functools import reduce
from typing import List, Dict

from demo.context import DispatchContext

from demo.dto import DispatchRequest, CourierPlan, ActionNode, Courier, Order
from demo.service import DispatchService
from nothing.app import toLocation

dispatchService = DispatchService()

def toDispatchRequest(dct):
    courierList = dct['couriers']
    courierobjs = []
    for courier in courierList:
        areaId = courier['areaId']
        id = courier['id']
        loc = courier['loc']
        maxLoads = courier['maxLoads']
        speed = courier['speed']
        courierobj = Courier(id, areaId, toLocation(loc), float(speed), int(maxLoads))
        courierobjs.append(courierobj)
    orderList = dct['orders']
    orderobjs = []
    for order in orderList:
        areaId = order['areaId']
        createTime = order['createTime']
        dstLoc = order['dstLoc']
        estimatedPrepareCompletedTime = order['estimatedPrepareCompletedTime']
        id = order['id']
        promiseDeliverTime = order['promiseDeliverTime']
        srcLoc = order['srcLoc']
        orderobj = Order(areaId, id, toLocation(srcLoc), toLocation(dstLoc), 0, int(createTime),
                         int(promiseDeliverTime), int(estimatedPrepareCompletedTime))
        orderobjs.append(orderobj)
    request = DispatchRequest(int(dct['requestTime']), dct['areaId'], bool(dct['firstRound']), bool(dct['lastRound']), courierobjs, orderobjs)
    return request


class DeliverEnv(object):
    def __init__(self, _areaId):
        # self.areaIds = ['680507', '725011', '730221']
        # self.dataFilePaths = ['data/' + fileName + '.json' for fileName in self.areaIds]
        # self.contexts: Dict[str, DispatchContext] = {}
        self.areaId = _areaId
        self.dataFilePath = './data/' + self.areaId + '.json'
        self._buildDeliver()
        self.doneTimeStamp = 1575694166
        self.haveRequest = False
        self.reward = 0
        self.requests = self.__readAllRequests()
        self.currentStep = 0

    def __readAllRequests(self):
        with open(self.dataFilePath, 'r') as r:
            # 第一行是开始的状态，在reset方法里会用，这里选择跳过
            lines = r.read().splitlines()[1:]
            return lines

    def getOneJsonRequest(self):
        index = self.currentStep % len(self.requests)
        if index == len(self.requests)-1:
            return json.loads(self.requests[index]), False
        else:
            return json.loads(self.requests[index]), True

    def _buildDeliver(self):
        self.context = DispatchContext(self.areaId, timeStamp=0)

    def reset(self):
        # 读第一个请求信息
        self.haveRequest = True
        json_request = None
        with open(self.dataFilePath, 'r') as r:
            line = r.readline()
            json_request = json.loads(line)
        # 处理请求
        dispatchRequest = toDispatchRequest(json_request)
        self.context.timeStamp = dispatchRequest.requestTimestamp
        self.context.addOnlineCouriers(dispatchRequest.couriers)
        self.context.addDispatchingOrders(dispatchRequest.orders)
        return self.context

    def step(self, courierActions: Dict[Courier, ActionNode]):
        # 得到当前状态
        s = self.context
        # 调度是否结束
        done = False
        allOrdersStats = [order.status == 4 for order in self.context.orderPool.orders]
        if reduce(lambda x, y: x & y, allOrdersStats):
            done = True
        # 是否还有新上线的骑手和订单
        if self.haveRequest:
            json_request, isHaveRequest = self.getOneJsonRequest()
            if not isHaveRequest:
                self.haveRequest = False
            # 更新骑手与订单状态
            dispatchRequest = toDispatchRequest(json_request)
            self.context.refresh(dispatchRequest.requestTimestamp)
            self.context.addOnlineCouriers(dispatchRequest.couriers)
            self.context.addDispatchingOrders(dispatchRequest.orders)
        else:
            # 更新时间戳
            self.context.timeStamp += 60
            self.context.refresh(self.context.timeStamp)

        # 奖励函数
        # 1.根据courierAction和时间改变当前的状态
        # 1.1 改变提交状态
        for courier, cp in courierActions.items():
            cp.setSubmitted(True)
        # 1.2 修改订单池中新单的状态
        allocatedOrders = list()
        for courier, cp in courierActions.items():
            if cp.actionType == 1:
                allocatedOrders.append(cp.orderId)
        self.context.markAllocatedOrders(allocatedOrders)
        # 1.3 根据courierActions修改骑手池的plan
        for courier, actionNode in courierActions.items():
            for courierInPool in self.context.courierPool.couriers:
                if courier.id == courierInPool.id:
                    courierInPool.planRoutes.append(actionNode)

        # 2.检查按时、超时、还未完成订单，分别给1分、-1分、0分
        self.reward = 0
        for order in self.context.orderPool.orders:
            if order.status != 4:
                break
            # 得到当前单的完成时间
            for courier in self.context.courierPool.couriers:
                for actionNode in courier.planRoutes:
                    if actionNode.orderId == order.id:
                        if actionNode.actionType == 3:
                            completeTime = actionNode.actionTime
                            if completeTime <= order.promiseDeliverTime:
                                self.reward += 1
                            else:
                                self.reward -= 1
        self.currentStep += 1
        # 下一步状态
        return self.context, self.reward, done

    def render(self):
        print(self.context.timeStamp, self.context.orderPool, self.reward)


