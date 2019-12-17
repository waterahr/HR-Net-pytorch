import numpy as np

def calculate_accuracy(gt_result, pt_result):
    ''' obtain the label-based and instance-based accuracy '''
    # compute the label-based accuracy
    if gt_result.shape != pt_result.shape:
        print('Shape beteen groundtruth and predicted results are different')
    # compute the label-based accuracy
    result = {}
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == -1).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == -1).astype(float) * (pt_result == -1).astype(float), axis=0)
    label_pos_acc = 1.0*pt_pos/(gt_pos + 1e-15)
    label_neg_acc = 1.0*pt_neg/(gt_neg + 1e-15)
    label_acc = (label_pos_acc + label_neg_acc)/2
    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_acc'] = label_acc
    result['label_ma'] = np.sum(label_acc) / len(label_acc)
    # compute the instance-based accuracy
    # precision
    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    floatersect_pos = np.sum((gt_result == 1).astype(float)*(pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1)+(pt_result == 1)).astype(float),axis=1)
    # avoid empty label in predicted results
    cnt_eff = float(gt_result.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff = cnt_eff - 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1
    instance_acc = np.sum(floatersect_pos/union_pos)/cnt_eff
    instance_precision = np.sum(floatersect_pos/pt_pos)/cnt_eff
    instance_recall = np.sum(floatersect_pos/gt_pos)/cnt_eff
    floatance_F1 = 2*instance_precision*instance_recall/(instance_precision+instance_recall)
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = floatance_F1
    return result

def find_best_fixed_threshold(preds, targs, do_plot=False):
    score = []
    thrs = np.arange(0, 0.6, 0.01)
    for thr in thrs:
        ### thr1/2
        #score.append(my_f2(targs, (preds > thr).astype(int) ))
        ### thr3
        score.append(mA_acc((preds[:, 0] > thr).astype(int), targs[:, 0]) )
    score = np.array(score)
    pm = score.argmax()
    best_thr, best_score = thrs[pm], score[pm].item()
    print('thr={best_thr:.3f}'.format(best_thr=best_thr), 'F2={best_score:.3f}'.format(best_score=best_score))
    if do_plot:
        plt.plot(thrs, score)
        plt.vlines(x=best_thr, ymin=score.min(), ymax=score.max())
        plt.text(best_thr+0.03, best_score-0.01, '$F_{2}=${best_score:.3f}'.format(best_score=best_score), fontsize=14);
        plt.show()
    return best_thr, best_score

def mA(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=-1)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=-1)
    true_negatives = np.sum(np.round(np.clip((1 - y_true) * (1 - y_pred), 0, 1)), axis=-1)
    possible_negatives = np.sum(np.round(np.clip(1 - y_true, 0, 1)), axis=-1)
    mean_acc = (true_positives / (possible_positives + sys.float_info.epsilon) + true_negatives / (possible_negatives + sys.float_info.epsilon)) / 2
    return mean_acc

def acc(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=-1)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=-1)
    possible_positives += np.sum(np.round(np.clip(y_pred, 0, 1)), axis=-1)
    possible_positives -= true_positives
    mean_acc = true_positives / (possible_positives + sys.float_info.epsilon)
    return np.average(mean_acc)

def prec(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=-1)
    possible_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=-1)
    mean_acc = true_positives / (possible_positives + sys.float_info.epsilon)
    return np.average(mean_acc)

def rec(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=-1)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=-1)
    mean_acc = true_positives / (possible_positives + sys.float_info.epsilon)
    return np.average(mean_acc)