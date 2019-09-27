import pandas as pd
import os
import math
import numpy as np

def stat_turn(path):
    ret = pd.DataFrame()
    for file_name in os.listdir(path):
        file_name = os.path.basename(file_name)
        file_name = file_name.split(".txt")[0]
        if file_name.endswith("out"):
            continue
        try:
            model, sp, heads, drop1, drop2, trs, acc = file_name.split("_")
        except:
            print(file_name)
            continue
        model = "_".join([model, sp, heads])

        ret.set_value(model, int(trs), float(acc))

    return ret

def stat_test(path, stat_range):
    results = []
    result_avg = None
    for i in stat_range:
        out_path = os.path.join(path, str(i))
        results.append(stat_turn(out_path))

    for r in results:
        if result_avg is None:
            result_avg = r.copy()
        else:
            result_avg += r
    result_avg /= len(stat_range)

    for r in results:
        r -= result_avg
        r = r.apply(lambda x: [abs(a) for a in x])
    result_err = np.stack([r.values for r in results], axis=-1)
    result_err = np.max(result_err, axis=-1)
    result_err = pd.DataFrame(result_err, result_avg.index, result_avg.columns)

    result_avg *= 100
    result_err *= 100
    result = pd.DataFrame(index=result_avg.index, columns=result_avg.columns)
    for i in result_avg.index:
        for j in result_avg.columns:
            result.set_value(i, j, "%.2f±%.2f"%(result_avg.loc[i, j], result_err.loc[i, j]))

    return result

if __name__ == "__main__":

    out_dir_path = "../out_cora"
    result = stat_test(out_dir_path, range(1, 10))
    result.to_csv("../cora_stat.csv", encoding="GBK")

    out_dir_path = "../out_cite"
    result = stat_test(out_dir_path, range(1, 10))
    result.to_csv("../citeseer_stat.csv", encoding="GBK")

    out_dir_path = "../out_pub"
    result = stat_test(out_dir_path, range(1, 10))
    result.to_csv("../pubmed_stat.csv", encoding="GBK")

