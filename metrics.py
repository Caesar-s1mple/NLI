from convert import labels_name

# test_seq1 = ['O', 'O', 'B-LocalID', 'I-LocalID', 'I-LocalID', 'O', 'B-LocalID', 'I-LocalID']
# test_seq2 = ['O', 'S-LocalID', 'O', 'B-LocalID', 'I-LocalID', 'I-LocalID', 'O', 'B-LocalID', 'I-LocalID', 'I-LocalID']
#
# true_seq = ['O', 'O', 'B-LocalID', 'I-LocalID', 'I-LocalID', 'O', 'S-LocalID', 'O']


def find_tag(tags, label):
    result = []
    for i, tag in enumerate(tags):
        if tag == 'S-' + label:
            result.append((i, 1))
        if tag == 'B-' + label:
            start_pos = i
        if tag == 'I-' + label and tags[i - 1] == 'B-' + label:
            length = 2
            for k in range(i + 1, len(tags)):
                if tags[k] == 'I-' + label:
                    length += 1
                else:
                    break
            result.append((start_pos, length))

    return result


def find_all_tags(tags):
    result = {}
    for label in labels_name:
        res = find_tag(tags, label)
        result[label] = res

    return result


def cal_precision(true_tags, pred_tags):
    pre = []
    pred_result = find_all_tags(pred_tags)
    if len(pred_result) == 0:
        return 0
    for tag, inf in pred_result.items():
        for start_pos, length in inf:
            if pred_tags[start_pos: start_pos + length] == true_tags[start_pos: start_pos + length]:
                pre.append(1)
            else:
                pre.append(0)

    return sum(pre) / len(pre)


def cal_recall(true_tags, pred_tags):
    recall = []
    pred_result = find_all_tags(true_tags)
    if len(pred_result) == 0:
        return 0
    for tag, inf in pred_result.items():
        for start_pos, length in inf:
            if pred_tags[start_pos: start_pos + length] == true_tags[start_pos: start_pos + length]:
                recall.append(1)
            else:
                recall.append(0)

    return sum(recall) / len(recall)


def score(true_tags, pred_tags):
    precision = cal_precision(true_tags, pred_tags)
    recall = cal_recall(true_tags, pred_tags)
    if recall == 0 or precision == 0:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall
