"""冻结父编码器的384维表示；计算顺序沿用选定父版，不保留旧读出参数。"""
import torch
from torch.nn import functional as F
from v3201_model import MultiFrequencyGRU
from v3201_schema import FREQUENCIES, prepare_model_inputs


class FrozenSharedEncoder(MultiFrequencyGRU):
    def __init__(self, config):
        super().__init__(config)
        del self.experts
        del self.router
        self.feature_dim = config['model']['fusion_hidden_dim']
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def fuse_features(self, vectors, clock):
        tokens = torch.stack(vectors, dim=1) + self.frequency_embedding
        tokens = self.frequency_fusion(tokens)
        return self.shared(torch.cat([tokens.flatten(1), clock], dim=1)).float()

    @torch.no_grad()
    def single_features(self, windows, clock):
        windows, lengths = prepare_model_inputs(windows, clock, self.window_bars)
        vectors = []
        for f in FREQUENCIES:
            raw = windows[f][:, :75, :]
            vector = self.encoders[f](raw, lengths[f])
            vectors.append(self.price_memories[f].single(vector, raw, windows[f][:, 75, :],
                                                        windows[f][:, 76, :] > 0, lengths[f]))
        return self.fuse_features(vectors, clock)

    @torch.no_grad()
    def group_features(self, windows, latest_clock):
        """返回八时点的冻结融合表示；不调用父收益读出。"""
        windows, _ = prepare_model_inputs(windows, latest_clock, self.window_bars)
        batch, device = latest_clock.shape[0], latest_clock.device
        vectors = []
        for frequency in FREQUENCIES:
            encoder = self.encoders[frequency]
            raw = windows[frequency][:, :75, :]
            states = encoder.intraday_states(raw)
            with torch.autocast(device_type=device.type, enabled=False):
                days, bars = encoder.days, encoder.daily_bars
                # 各频率的历史日先跨日编码一次，八个时点从同一历史状态分别接入今日前缀。
                previous, history_state = encoder.interday(encoder.dropout(states[:, :-1, -1]))
                last = (torch.arange(8, device=device) * (bars // 8) - 1).clamp_min(0)
                current = states[:, -1, last].transpose(0, 1).clone()
                current[0] = 0
                updated, _ = encoder.interday(encoder.dropout(current).reshape(8 * batch, 1, -1),
                                               history_state.repeat(1, 8, 1))
                updated = updated[:, 0].reshape(8, batch, -1)
                available_today = (torch.arange(8, device=device) > 0)[:, None, None]
                vector = torch.where(available_today, updated, previous[:, -1].unsqueeze(0))
            query_lengths = ((days - 1) * bars + torch.arange(8, device=device) * (bars // 8))[None, :].expand(batch, -1)
            vector = self.price_memories[frequency](vector.transpose(0, 1), raw,
                        windows[frequency][:, 75, :], windows[frequency][:, 76, :] > 0, query_lengths)
            vectors.append(vector.transpose(0, 1).reshape(8 * batch, -1))
        clocks = torch.eye(8, device=device, dtype=latest_clock.dtype).repeat_interleave(batch, dim=0)
        return self.fuse_features(vectors, clocks).reshape(8, batch, -1).transpose(0, 1)
