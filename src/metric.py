import argparse
import json


def flat_data(data):
    if 'turn_idx' in data[0] and 'turn_idx' in data[0]:
        return data
    assert 'dialogue' in data[0] and \
        'turn_idx' in data[0]['dialogue'][0] and \
        'belief_state' in data[0]['dialogue'][0]
    data_flat = []
    for dialog in data:
        for turn in dialog['dialogue']:
            item = {}
            data_flat.append(item)
            item['dialogue_idx'] = dialog['dialogue_idx']
            item['turn_idx'] = turn['turn_idx']
            item['belief_state'] = turn['belief_state']
    return data_flat


def metric(pred_data, truth_data):
    pred_data = flat_data(pred_data)
    truth_data = flat_data(truth_data)

    assert len(pred_data) == len(truth_data) > 0
    pred_data.sort(key=lambda turn: (turn['dialogue_idx'], turn['turn_idx']))
    truth_data.sort(key=lambda turn: (turn['dialogue_idx'], turn['turn_idx']))
 
    total, joint_acc, f1 = 0, 0.0, 0.0
    for pred_turn, turth_turn in zip(pred_data, truth_data):
        assert pred_turn['dialogue_idx'] == turth_turn['dialogue_idx']
        assert pred_turn['turn_idx'] == turth_turn['turn_idx']
        total += 1

        pred = pred_turn['pred_belief_state']
        truth = turth_turn['belief_state']

        # Filter none value
        pred = set(s for s in pred if s.split('-')[-1] != 'none')
        truth = set(s for s in truth if s.split('-')[-1] != 'none')

        # Joint ACC
        if pred == truth:
            joint_acc += 1

        # F1
        if len(truth) == 0 and len(pred) == 0:
            turn_f1 = 1.0
        else:
            tp = sum(s in pred for s in truth)
            fn = sum(s not in pred for s in truth)
            fp  = sum(s not in truth for s in pred)
            p = tp / (tp + fp) if tp + fp != 0 else 0.0
            r = tp / (tp + fn) if tp + fn != 0 else 0.0
            turn_f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0.0
        f1 += turn_f1

    joint_acc = joint_acc / total
    f1 = f1 / total
    scores = {
        'joint_acc': joint_acc * 100,
        'f1': f1 * 100,
    }
    return scores


def metric_file(pred_path, truth_path):
    with open(pred_path) as f:
        pred = json.load(f)
    with open(truth_path) as f:
        truth = json.load(f)
    return metric(pred_data=pred, truth_data=truth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', '-p', required=True)
    parser.add_argument('--truth_path', '-t', required=True)
    args = parser.parse_args()

    scores = metric_file(
        pred_path=args.pred_path,
        truth_path=args.truth_path,
    )

    for k, v in scores.items():
        print(f'{k}: {v:.2f}')


'''
python src/metric.py \
    -p outputs/multi/version_1/results/pred-valid_cnt=1.json \
    -t data/clean/valid.json

joint_acc: 49.08
f1: 89.59
'''
