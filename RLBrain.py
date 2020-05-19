# -*- coding: utf-8 -*-
# @Author  : woleto
# @Time    : 2020/5/11 10:46
import random
from typing import List, Dict

import dgl
from dgl import graph, DGLGraph
import torch
from dgl.nn import GraphConv
from torch import nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from demo.dto import Courier, ActionNode
from demo.util import DistanceUtils



class GCN(nn.Module):
    def __init__(self):
        in_dim = 2
        hidden_dim = 256
        n_classes = 2
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        # 二分类，以第一个数为合理的值。也就是输出形状为（2），第一个为合理的值
        self.classify = nn.Linear(hidden_dim, n_classes)
        # self.classify.weight.data.normal_(0, 0.1)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        # expect_time = g.ndata['expect_time'].view(-1, 1).float()
        # promise_deliver_time = g.ndata['promise_deliver_time'].view(-1, 1)
        # courier_level = g.ndata['courier_level'].view(-1, 1)
        # courier_speed = g.ndata['courier_speed'].view(-1, 1)
        # h = np.stack((expect_time, promise_deliver_time, courier_level, courier_speed),1).reshape(-1, 4)
        # h = torch.as_tensor(h)
        # 加入节点特征
        action_Time = g.ndata['action_Time'].view(-1,1).float()
        action_Type = g.ndata['action_Type'].view(-1,1).float()
        h = np.stack((action_Time, action_Type), 1).reshape(-1, 2)
        h = torch.as_tensor(h)

        # h = g.in_degrees().view(-1, 1).float()
        # print(h.shape)
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        # print(hg.shape, hg)
        # print('*'*50)
        # hg = g.ndata['h'][g.number_of_nodes() - 1]
        # print(self.classify(hg), self.classify(hg).shape)
        return self.classify(hg)


class Transition:
    def __init__(self, _observation, _actions, _reward, _observation_):
        self.observation = _observation
        self.actions = _actions
        self.reward = _reward
        self.observation_ = _observation_


class DeepQNetwork(object):
    def __init__(self):
        # DQN需要使用两个神经网络
        self.eval_net, self.target_net = GCN(), GCN()
        self.learn_step_counter = 0  # 用于 target 更新计时，100次更新一次
        self.memory_counter = 0  # 记忆库记数
        self.memory_size = 500
        self.replace_target_iter = 300
        self.batchSize = 16
        self.gamma = 0.9
        self.epsilon = 0.9
        self.memory: List[Transition] = [Transition(None, None, None, None) for _ in range(self.memory_size)]
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)  # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式

    def choose_action(self, observation):
        actions: Dict[Courier, ActionNode] = {}
        # Note：返回的courierObservationGraphs中，骑手的图列表有可能为空
        courierCandidateActions, courierObservationGraphs = self.__calGraphsByObservation(observation)
        if np.random.randn() <= self.epsilon:  # greedy policy
            # 找到每个骑手候选动作中值最大的动作
            for courier, graphs in courierObservationGraphs.items():

                batchGraphs = dgl.batch(graphs)
                action_value = self.eval_net.forward(batchGraphs)
                # 找出哪一个最大
                maxIndex = action_value[:, 0].argmax(0)

                actions[courier] = courierCandidateActions.get(courier)[maxIndex]
        else:  # random policy
            # 在候选动作中随机选择一个动作，构造ActionNode类进行返回
            for courier, actionNodes in courierCandidateActions.items():
                actionNode = random.choice(actionNodes)
                actions[courier] = actionNode
        return actions

    def store_transition(self, observation, actions, reward, observation_):
        transition = Transition(observation, actions, reward, observation_)
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter += 1

    def learn(self):
        # update the parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        transition_index = [int(x) for x in np.random.choice(self.memory_size, self.batchSize)]

        # q_eval
        # 根据observation构成批次的图，由于一个observation就是很多张图，很多张图只对应一个reward，先一个observation一个训练
        batchReward = []
        batchQEval = []
        batchQNext = []
        for index in transition_index:
            memory = self.memory[index]
            if memory.observation is None:
                continue
            # 0. 将当前observation对应的reward加入batchReward
            batchReward.append(memory.reward)
            # 针对一个observation，actions， reward以及observation_计算结果
            # 1.1 使用observation与actions信息构造图，从observation得到每一个骑手的历史节点，从actions得到要加入的新的节点，合起来作为一个图
            _, courierObservationGraphs = self.__calGraphsByObservation(memory.observation, actions=memory.actions)
            # 1.2 将1.1中的所有图放入eval_net，得到预测的合理值（1，）
            # courierObservationBatchGraphs = dgl.batch(map(lambda x, y: x + y, courierObservationGraphs.values()))
            courierObservationBatchGraphs = []
            for graphs in courierObservationGraphs.values():
                courierObservationBatchGraphs.append(graphs[0])
            courierObservationBatchGraphs = dgl.batch(courierObservationBatchGraphs)
            courierQEval = self.eval_net(courierObservationBatchGraphs)[:, 0]
            courierQEval = torch.sum(torch.Tensor(courierQEval))
            # 1.3 将预测值求和加入batchQEval
            batchQEval.append(courierQEval)

            # 2.1 使用observation_信息构造图，从observation_得到每一个骑手的历史节点，所有动作节点分别加进去，合起来构图
            _, courierObservation_Graphs = self.__calGraphsByObservation(memory.observation_)
            # 2.2 将2.1中所有图放入target_net，得到预测的合理值
            # 2.3 找到每个骑手在2.1中生成的图预测合理值最大的那个值
            # 2.4 最大值求和加入batchQNext
            # 所有骑手的图最大合理值
            courierMaxGraphValues = []
            for courier, graphs in courierObservation_Graphs.items():
                courierBatchGraphs = dgl.batch(graphs)
                courierQNext = self.target_net(courierBatchGraphs)
                courierQNext = courierQNext[:, 0].max(0)
                courierMaxGraphValues.append(courierQNext)
            courierQNext = torch.sum(torch.Tensor(courierMaxGraphValues))
            batchQNext.append(courierQNext)

        # 3.计算qtarget和loss
        q_target = torch.Tensor(np.array(batchReward) + self.gamma * np.array(batchQNext)).detach()
        q_target.requires_grad = True
        q_eval = torch.Tensor(batchQEval).detach()
        loss = self.loss_func(q_eval, q_target)
        # print(q_eval, q_target, loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self.target_net

    def __calActionNodeTime(self, courier, order, currentStats, context):
        """
        通过courier找到上一个动作位置和时间，通过order与currentStats计算将要执行的动作的位置
        :param courier:
        :param order:
        :param currentStats:
        :return:
        """
        distanceUtils = DistanceUtils()
        if len(courier.planRoutes) == 0 or courier.planRoutes[-1].actionType == -1:
            calActionloc1 = order.srcLoc
            calActionloc2 = order.dstLoc
            lng1 = courier.loc.longitude
            lat1 = courier.loc.latitude
            if currentStats == 1:
                lng2 = calActionloc1.longitude
                lat2 = calActionloc1.latitude
                calDistance = distanceUtils.greatCircleDistance(lng1, lat1, lng2, lat2)
                calTime = calDistance / courier.speed
                # lastActionTime + 所需时间
                calActionTime = context.timeStamp + calTime
            elif currentStats == 2:
                # 计算距离，除速度得到所需时间，如果这一步是取餐（2），则要在订单做出来之后
                lng2 = calActionloc1.longitude
                lat2 = calActionloc1.latitude
                calDistance = distanceUtils.greatCircleDistance(lng1, lat1, lng2, lat2)
                calTime = calDistance / courier.speed
                # 骑手到店时间 = 最后一个动作完成的时间+所需的时间
                courier_arriveTime = context.timeStamp + calTime
                if courier_arriveTime < order.estimatedPrepareCompletedTime:
                    calActionTime = order.estimatedPrepareCompletedTime
                else:
                    calActionTime = courier_arriveTime
            else:
                lng2 = calActionloc2.longitude
                lat2 = calActionloc2.latitude
                calDistance = distanceUtils.greatCircleDistance(lng1, lat1, lng2, lat2)
                calTime = calDistance / courier.speed
                calActionTime = context.timeStamp + calTime
            return int(calActionTime)
        else:
            lastActionNode = courier.planRoutes[-1]
            lastActionTime = lastActionNode.actionTime  # 最后一个动作得时间
            # 得到最后一个动作所在的位置
            lastActionType = lastActionNode.actionType
            order_info = [i for i in context.orderPool.orders if lastActionNode.orderId == i.id]
            if order_info:
                if lastActionType == 1 or lastActionType == 2:
                    if order_info:
                        # 取到最后一个动作所在订单得取货地
                        lastActionloc = order_info[0].srcLoc
                    else:
                        # 取到最后一个动作所在订单得送货地
                        lastActionloc = order_info[0].dstLoc
                    # 得到要计算的动作所在的位置
                    calActionloc1 = order.srcLoc
                    calActionloc2 = order.dstLoc
                    lng1 = lastActionloc.longitude
                    lat1 = lastActionloc.latitude
                    if currentStats == 1 :
                        lng2 = calActionloc1.longitude
                        lat2 = calActionloc1.latitude
                        calDistance = distanceUtils.greatCircleDistance(lng1, lat1, lng2, lat2)
                        calTime = calDistance / courier.speed
                        # lastActionTime + 所需时间
                        calActionTime = lastActionTime + calTime
                    elif currentStats == 2:
                        # 计算距离，除速度得到所需时间，如果这一步是取餐（2），则要在订单做出来之后
                        lng2 = calActionloc1.longitude
                        lat2 = calActionloc1.latitude
                        calDistance = distanceUtils.greatCircleDistance(lng1, lat1, lng2, lat2)
                        calTime = calDistance / courier.speed
                        # 骑手到店时间 = 最后一个动作完成的时间+所需的时间
                        courier_arriveTime = lastActionTime + calTime
                        if courier_arriveTime < order.estimatedPrepareCompletedTime:
                            calActionTime = order.estimatedPrepareCompletedTime
                        else:
                            calActionTime = courier_arriveTime
                    else:
                        lng2 = calActionloc2.longitude
                        lat2 = calActionloc2.latitude
                        calDistance = distanceUtils.greatCircleDistance(lng1, lat1, lng2, lat2)
                        calTime = calDistance / courier.speed
                        calActionTime = lastActionTime + calTime
                    return int(calActionTime)

    def __calGraphsByObservation(self, observation, actions: Dict[Courier, ActionNode] = None):
        """
        待解决的问题： 2020-5-12 20:35:34
        1.生成节点时没有计算动作完成时间戳,考虑取餐时间戳要大于制作完成的时间
        2.节点构图
        Note:
            1.返回的Dict[Courier, List[graph]]， 列表有可能为空，代表没有候选动作。没有设计什么也不做的动作。
        :param batch_observation:
        :return:
        """

        courierCandidateActions: Dict[Courier, List[ActionNode]] = {}
        courierGraphs: Dict[Courier, List[graph]] = {}

        if actions is None:
            # 1.循环处理骑手与订单信息，计算每个骑手下一步的候选动作，将计算结果保存在courierCandidateActions中
            for courier in observation.courierPool.couriers:
                courierCandidateActions[courier] = list()
                # 如果最后一个执行动作没有做完，则添加一个actionType为-1的actionNode，代表什么也不做
                if len(courier.planRoutes) > 0:
                    if courier.planRoutes[-1].actionType != -1:
                        if courier.planRoutes[-1].actionTime < observation.timeStamp:
                            candidateActionNode = ActionNode(-1, None, -1, None, None)
                            courierCandidateActions.get(courier).append(candidateActionNode)
                            continue
                for order in courier.orders:
                    if order.status == 4 or order.status == 3:
                        continue
                    elif order.status == 2:
                        actionTypeTime = self.__calActionNodeTime(courier, order, 3, observation)
                        candidateActionNode = ActionNode(3, order.id, actionTypeTime, None, None)
                        courierCandidateActions.get(courier).append(candidateActionNode)
                    elif order.status == 1:
                        actionTypeTime = self.__calActionNodeTime(courier, order, 2, observation)
                        candidateActionNode = ActionNode(2, order.id, actionTypeTime, None, None)
                        courierCandidateActions.get(courier).append(candidateActionNode)
            # 2.1 得到待分配的新单
            orderPoolOrders = observation.orderPool.orders
            newOrders = []
            for order in orderPoolOrders:
                if order.status == 0:
                    newOrders.append(order)
            # 2.2 将新单的到店动作加到每一个骑手的候选动作中
            for newOrder in newOrders:
                # map(lambda courier, actionNodes: actionNodes.append(ActionNode(1, newOrder.id, self.__calActionNodeTime(courier, order, 1))), courierCandidateActions.keys(), courierCandidateActions.values())
                for courier, actionNodes in courierCandidateActions.items():
                    addActionNode = ActionNode(1, newOrder.id, self.__calActionNodeTime(courier, newOrder, 1, observation), None, None)
                    actionNodes.append(addActionNode)

            # 3. 如果没有动作可以做，则添加一个什么也不做的动作
            for courier, actions in courierCandidateActions.items():
                if len(actions) == 0:
                    candidateActionNode = ActionNode(-1, None, -1, None, None)
                    actions.append(candidateActionNode)
        else:
            for courier, actionNode in actions.items():
                courierCandidateActions[courier] = [actionNode]

        # 2.1 使用courierCandidateActions构造图
        for courier, actionNodes in courierCandidateActions.items():
            courierGraphs[courier] = list()
            historyNodes = courier.planRoutes
            for actionNode in actionNodes:
                # 处理actionNode中 actionType为-1 的情情况，方式为：如果为-1，则不添加该节点，而只是把历史节点构图
                currentNodes = historyNodes
                if actionNode.actionType > -1:
                    currentNodes.append(actionNode)
                # Todo
                # 使用所有的节点构图
                currentGraph = self.__generateGraph(courier, currentNodes)

                # 纯测试代码，请无视
                # currentGraph = DGLGraph()
                # currentGraph.add_nodes(4)
                # currentGraph.add_edges([0, 1, 2], [1, 2, 3])
                # currentGraph.ndata['h'] = np.array([1, 2, 3, 4])

                courierGraphs.get(courier).append(currentGraph)

        return courierCandidateActions, courierGraphs,

    def __generateGraph(self, courier: Courier, currentNodes):
        """
        目标：已有骑手已完成的动作，依次加入可以加入的节点，形成若干图，放进图网络，得到合理值。
        :param couriers: 所有骑手的信息
        :param courierPlans: 所有骑手已完成的动作
        :param Orders: 订单 包括已完成的单和新单
        :return:
        """
        courier_list = []
        # 骑士id
        id = courier.id  # id: 1604576185
        # 骑士所在商圈id
        areaId = courier.areaId  # areaId: 680507
        # 骑士所在经度 纬度
        courier_latitude = courier.loc.latitude  # lat: 40.468577
        courier_longitude = courier.loc.longitude # long: 102.721584
        # 骑士速度
        speed = courier.speed  # speed: 3.068057263223962
        # 骑士最大载单量
        maxloads = courier.maxLoads  # maxloads: 10
        # 骑士已完成的路径
        planRoutes = currentNodes
        # 骑士的订单
        orders = courier.orders
        courier_list.extend([id, areaId, courier_latitude, courier_longitude, speed,
                             maxloads, planRoutes, orders])
        # graph_name = id
        graph = self.__calculate_graph(courier_list)
        return graph

    # 生成图
    def __calculate_graph(self, courier_list):
        # 图初始化
        # 一个骑士的订单
        id = courier_list[0]
        areaId = courier_list[1]
        speed = courier_list[4]
        maxloads = courier_list[5]
        orders = courier_list[-1]
        planRoutes = courier_list[-2]
        graph = dgl.DGLGraph()

        # 添加节点
        graph.add_nodes(len(planRoutes))

        # 添加边
        for index in range(len(planRoutes)-1):
            if planRoutes[index].actionType == 1:
                graph.add_edge(index, index + 1)
                distance = np.array(list([1])).astype('float32')
                graph.edges[index, index+1].data['distance'] = distance
            elif planRoutes[index].actionType == 2:
                order_info = [i for i in orders if i.id == planRoutes[index].orderId]
                lng1 = order_info[0].srcLoc.longitude
                lat1 = order_info[0].srcLoc.latitude
                lng2 = order_info[0].dstLoc.longitude
                lat2 = order_info[0].dstLoc.latitude
                distance = DistanceUtils.greatCircleDistance(lng1, lat1, lng2, lat2)
                distance = np.array(list([distance])).astype('float32')
                graph.add_edge(index, index + 1)
                graph.edges[index, index+1].data['distance'] = distance
            else:
                # 上一个订单送达
                order_info = [i for i in orders if i.id == planRoutes[index].orderId]
                # 下一个订单取餐
                order_info1 = [i for i in orders if i.id == planRoutes[index+1].orderId]
                if order_info and order_info1:
                    lng1 = order_info[0].dstLoc.longitude
                    lat1 = order_info[0].dstLoc.latitude
                    lng2 = order_info1[0].srcLoc.longitude
                    lat2 = order_info1[0].srcLoc.latitude
                    distance = DistanceUtils.greatCircleDistance(lng1, lat1, lng2, lat2)
                    distance = np.array(list([distance])).astype('float32')
                    graph.add_edge(index, index + 1)
                    graph.edges[index, index + 1].data['distance'] = distance
        # 把动作节点时间和类型加入节点属性中
        action_Time_list = [planRoutes[index].actionTime for index in range(len(planRoutes))]
        action_Type_list = [planRoutes[index].actionType for index in range(len(planRoutes))]
        action_Time = np.array(action_Time_list).astype('float32')
        action_Type = np.array(action_Type_list).astype('float32')
        graph.ndata['action_Time'] = action_Time
        graph.ndata['action_Type'] = action_Type

        return graph