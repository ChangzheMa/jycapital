"""只允许显式授权的协议标识与选模指标差异，保留完整评估契约。"""

SELECTION_FIELDS = frozenset(("protocol_id", "selection_metric"))


def validate_protocol_compatibility(left, right, *, allow_selection_metric_change=False):
    differences = {key: {"left": left.get(key), "right": right.get(key)}
                   for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)
                   or (key in left) != (key in right)}
    if differences and (not allow_selection_metric_change or not set(differences) <= SELECTION_FIELDS):
        raise ValueError("评估契约不兼容: " + ", ".join(differences))
    return differences
