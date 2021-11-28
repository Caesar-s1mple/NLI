from sklearn.metrics import f1_score, precision_score, recall_score


def score(true_tags, pred_tags):
    return f1_score(true_tags, pred_tags, average="macro"), \
           precision_score(true_tags, pred_tags, average="macro"), \
           recall_score(true_tags, pred_tags, average="macro")
