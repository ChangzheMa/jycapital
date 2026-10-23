"""冻结父网络的完整因果融合表示与当前原输入，保持股票日共享计算。"""
import torch

from v2401_model import MultiFrequencyGRU
from v2401_schema import FREQUENCIES, TRADING_DAY_WINDOWS, prepare_model_inputs


class FrozenHistoryEncoder(MultiFrequencyGRU):
    def __init__(self, parent_config):
        super().__init__(parent_config)
        self.feature_dim = parent_config['model']['fusion_hidden_dim'] + 3 * 77 + 8 + 1
        self._captured_features = None
        # 参数不交给优化器；保持父模型的grad标志以沿用相同CPU推理算子。
        # 正式特征提取使用inference_mode，权重在提取前后核对指纹。
        self.eval()

    def fuse_components(self, vectors, clock):
        # 与冻结父源码相同的计算顺序，只额外返回已计算的融合表示。
        tokens = torch.stack(vectors, dim=1) + self.frequency_embedding
        tokens = self.frequency_fusion(tokens)
        shared = self.shared(torch.cat([tokens.flatten(1), clock], dim=1))
        experts = torch.stack([expert(shared) for expert in self.experts], dim=1).float()
        logits = self.router(shared).float()
        probabilities = torch.softmax(logits, dim=-1)
        self._captured_features = shared
        return (experts * probabilities[..., None]).sum(dim=1), experts, logits

    def _latest_inputs(self, windows, clock_ids):
        values = []
        batch = next(iter(windows.values())).shape[0]
        for frequency in FREQUENCIES:
            bars = 240 // int(frequency[3:-1])
            count = TRADING_DAY_WINDOWS[frequency]['previous_days'] * bars + clock_ids * (bars // 8)
            index = (count - 1).clamp_min(0)
            source = windows[frequency]
            selected = source.gather(2, index[:, None, :].expand(batch, 77, -1)).transpose(1, 2)
            values.append(selected)
        return torch.cat(values, dim=-1)

    def group_features(self, windows, latest_clock):
        if self.training:
            raise ValueError('树的历史编码器必须保持冻结推理状态')
        windows, _ = prepare_model_inputs(windows, latest_clock, self.window_bars)
        self._captured_features = None
        try:
            scores = super().forward_group(windows, latest_clock)
            batch = latest_clock.shape[0]
            shared = self._captured_features.reshape(8, batch, -1).transpose(0, 1)
            ids = torch.arange(8, device=latest_clock.device)[None, :].expand(batch, -1)
            clocks = torch.eye(8, device=latest_clock.device)[None, :, :].expand(batch, -1, -1)
            primary = scores[..., 0]
            features = torch.cat((shared, self._latest_inputs(windows, ids), clocks, primary[..., None]), dim=-1).float()
            return features, primary
        finally:
            self._captured_features = None

    def single_features(self, windows, clock):
        if self.training:
            raise ValueError('树的历史编码器必须保持冻结推理状态')
        windows, _ = prepare_model_inputs(windows, clock, self.window_bars)
        self._captured_features = None
        try:
            primary = super().forward_heads(windows, clock)[:, 0]
            raw = self._latest_inputs(windows, clock.argmax(dim=1)[:, None])[:, 0]
            return torch.cat((self._captured_features, raw, clock, primary[:, None]), dim=-1).float(), primary
        finally:
            self._captured_features = None
