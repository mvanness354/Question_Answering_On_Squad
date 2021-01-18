import numpy as np

"""
I am making the assumption that a_gold and a_pred are lists of tuples
where each tuple is (start_index, end_index).  I'm also assuming that
a tuple of (0, 0) represents a "no answer" prediction.  This setup is compatible
with both of my model implementations (following @1233 on Piazza).  Note that since predicting no answer is represented by the tuple (0, 0),
f1 will be 1 already when both a_pred and a_gold are (0, 0) since precision
and recall will both be 1
"""


def tuple_overlap(x, y):
  return max(min(x[1], y[1]) - max(x[0], y[0]) + 1, 0)

def compute_exact(a_gold, a_pred):
  return sum( (x == y for x, y in zip(a_gold, a_pred)) ) / len(a_gold)

def compute_f1(a_gold, a_pred):

  scores = []
  for pred, known in zip(a_pred, a_gold):
    overlap = tuple_overlap(pred, known)
    num_pred = pred[1] - pred[0] + 1
    num_known = known[1] - known[0] + 1
    precision = overlap / num_pred
    recall = overlap / num_known

    if precision + recall > 0:
      f1 = (2*precision*recall) / (precision + recall)
    else:
      f1 = 0

    scores.append(f1)

  return np.mean(scores), scores