"""完整预测时点内标准化后等权组合；不读取标签。"""

import numpy as np
import pandas as pd

KEYS = ['date', 'instrument']


def combine_scores(keys, scores):
    if list(keys.columns) != KEYS or keys[KEYS].isna().any().any() or keys.duplicated(KEYS).any():
        raise ValueError('组合要求唯一、非空的 date/instrument 键')
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(keys), 2) or not np.isfinite(scores).all():
        raise ValueError('组合要求两列有限、完整对齐的分数')
    result = keys.copy().reset_index(drop=True)
    if not len(result):
        result['factor'] = np.array([], dtype=np.float64)
        return result
    values = pd.DataFrame(scores)
    centered = values - values.groupby(result.date).transform('mean')
    scale = centered.pow(2).groupby(result.date).transform('mean').pow(.5)
    normalized = centered.div(scale.where(scale > 0)).fillna(0)
    result['factor'] = normalized.mean(axis=1).to_numpy(dtype=np.float64)
    if not np.isfinite(result.factor).all():
        raise FloatingPointError('组合输出非有限')
    return result


def align_components(left, right):
    for frame in (left, right):
        if list(frame.columns) != KEYS + ['factor'] or frame.duplicated(KEYS).any():
            raise ValueError('组件预测格式或键不合法')
    aligned = left.merge(right, on=KEYS, how='outer', suffixes=('_left', '_right'), validate='one_to_one')
    scores = aligned[['factor_left', 'factor_right']].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('两个组件预测必须完整、有限且键相同')
    return aligned[KEYS], scores
