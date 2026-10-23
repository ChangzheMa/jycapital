"""将标量数值分裂的XGBoost树转换为可独立推理的Torch数组。"""
import json
import math

import torch
from torch import nn


class TreeEnsemble(nn.Module):
    def __init__(self, features, max_trees, max_depth, tree_batch_size=128):
        super().__init__()
        if min(features, max_trees, max_depth, tree_batch_size) < 1:
            raise ValueError('树布局必须为正整数')
        self.features, self.max_trees, self.max_depth = features, max_trees, max_depth
        self.tree_batch_size = tree_batch_size
        shape = (max_trees, 2 ** (max_depth + 1) - 1)
        self.register_buffer('left', torch.full(shape, -1, dtype=torch.int32))
        self.register_buffer('right', torch.full(shape, -1, dtype=torch.int32))
        self.register_buffer('feature', torch.zeros(shape, dtype=torch.int32))
        self.register_buffer('condition', torch.zeros(shape, dtype=torch.float32))
        self.register_buffer('default_left', torch.zeros(shape, dtype=torch.bool))
        self.register_buffer('active_trees', torch.zeros((), dtype=torch.int64))

    @torch.no_grad()
    def import_booster(self, document):
        document = json.loads(document) if isinstance(document, (str, bytes, bytearray)) else document
        learner = document['learner']
        parameter = learner['learner_model_param']
        base = parameter['base_score']
        base = json.loads(base) if isinstance(base, str) else base
        base = base[0] if isinstance(base, list) and len(base) == 1 else base
        if float(base) != 0. or parameter.get('boost_from_average', '0') != '0':
            raise ValueError('父预测作为固定初始值时，树自身初始值必须为零')
        if int(parameter['num_feature']) != self.features or int(parameter.get('num_target', 1)) != 1 or int(parameter['num_class']) != 0:
            raise ValueError('只支持当前维度的标量回归树')
        if learner['objective']['name'] != 'reg:squarederror':
            raise ValueError('树输出要求未变换的平方误差回归分数')
        booster = learner['gradient_booster']
        if booster['name'] != 'gbtree':
            raise ValueError('只支持普通梯度提升树')
        model = booster['model']
        trees = model['trees']
        if len(trees) > self.max_trees or model['tree_info'] != [0] * len(trees):
            raise ValueError('树数量或分组不符合声明')
        if model.get('iteration_indptr') != list(range(len(trees) + 1)):
            raise ValueError('每次迭代必须只有一棵标量树')
        checked = []
        for tree in trees:
            size = int(tree['tree_param']['num_nodes'])
            names = ('left_children','right_children','split_indices','split_conditions','default_left','split_type')
            if size < 1 or size > self.left.shape[1] or any(len(tree[n]) != size for n in names):
                raise ValueError('树节点布局不完整或超过预算')
            if int(tree['tree_param'].get('num_deleted', 0)) != 0 or any(tree['split_type']):
                raise ValueError('不支持删除节点或类别分裂')
            visited, stack = set(), [(0, 0)]
            while stack:
                index, depth = stack.pop()
                if index in visited or not 0 <= index < size or depth > self.max_depth:
                    raise ValueError('树存在循环、越界或超过声明深度')
                visited.add(index)
                left, right = tree['left_children'][index], tree['right_children'][index]
                if not math.isfinite(tree['split_conditions'][index]):
                    raise ValueError('分裂阈值或叶值非有限')
                if left == -1 and right == -1:
                    continue
                if left < 0 or right < 0 or not 0 <= tree['split_indices'][index] < self.features:
                    raise ValueError('内部节点的子节点或特征不合法')
                stack.extend(((left, depth + 1), (right, depth + 1)))
            if len(visited) != size:
                raise ValueError('树包含不可达节点')
            checked.append(tree)
        # 在CPU一次组装，再按数组传输，避免每棵树反复进行小型CUDA分配。
        names = ('left','right','feature','condition','default_left')
        packed = {name:torch.empty_like(getattr(self,name),device='cpu') for name in names}
        packed['left'].fill_(-1)
        packed['right'].fill_(-1)
        for name in ('feature','condition','default_left'):packed[name].zero_()
        for number, tree in enumerate(checked):
            size = len(tree['left_children'])
            for target, source in (('left','left_children'),('right','right_children'),('feature','split_indices'),
                                   ('condition','split_conditions'),('default_left','default_left')):
                destination = packed[target]
                destination[number, :size].copy_(torch.tensor(tree[source], dtype=destination.dtype))
        for name in names:getattr(self,name).copy_(packed[name])
        self.active_trees.fill_(len(checked))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.features or features.device != self.condition.device:
            raise ValueError('树输入维度或设备不符')
        if torch.isinf(features).any():
            raise ValueError('树输入不能含无穷值')
        result = torch.zeros(features.shape[0], dtype=torch.float32, device=features.device)
        count = int(self.active_trees.item())
        if not 0 <= count <= self.max_trees:
            raise ValueError('有效树数越界')
        rows = torch.arange(len(features), device=features.device)[:, None]
        for start in range(0, count, self.tree_batch_size):
            trees = torch.arange(start, min(start + self.tree_batch_size, count), device=features.device)[None, :]
            nodes = torch.zeros((len(features), trees.numel()), dtype=torch.long, device=features.device)
            for _ in range(self.max_depth):
                left, right = self.left[trees, nodes], self.right[trees, nodes]
                leaves = left < 0
                value = features[rows, self.feature[trees, nodes].long()]
                goes_left = torch.where(torch.isnan(value), self.default_left[trees, nodes], value < self.condition[trees, nodes])
                next_node = torch.where(goes_left, left, right).long()
                nodes = torch.where(leaves, nodes, next_node)
            result = result + self.condition[trees, nodes].sum(dim=1)
        return result
