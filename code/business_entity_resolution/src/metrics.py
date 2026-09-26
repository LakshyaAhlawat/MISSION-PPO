"""F_beta macro-average scoring, matching the challenge's evaluation exactly."""


def f_beta_macro(y_true, y_pred, all_ids, beta=0.5):
    """y_true, y_pred: dict[str, set[str]] keyed by source1_entity_id.
    all_ids: iterable of every source1_entity_id in the evaluation set
    (entities missing from y_true/y_pred are treated as empty sets)."""
    beta2 = beta * beta
    scores = []
    for eid in all_ids:
        true_set = y_true.get(eid, set())
        pred_set = y_pred.get(eid, set())

        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue

        tp = len(true_set & pred_set)
        precision = tp / len(pred_set)
        recall = tp / len(true_set)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((1 + beta2) * precision * recall / (beta2 * precision + recall))

    return sum(scores) / len(scores) if scores else 0.0
