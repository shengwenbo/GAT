from utils.process import load_data
import numpy as np

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")

num_nbs = np.sum(adj, axis=-1).tolist()
list.sort(num_nbs, reverse=True)
print(num_nbs)