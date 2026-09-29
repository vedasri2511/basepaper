import numpy as np

def calculate_ac_k(predicted_scores, ground_truth_indices, k_list=[1, 3, 5]):
    """
    Section V.B.3: Evaluation Metrics (AC@k)
    Calculates AC@k for resource entity localization (Dim 1) and 
    observability feature identification (Dim 2).
    """
    results = {}
    top_indices = np.argsort(predicted_scores)[::-1]
    
    for k in k_list:
        top_k = set(top_indices[:k])
        hit = len(top_k.intersection(ground_truth_indices)) > 0
        results[f'AC@{k}'] = 1.0 if hit else 0.0
        
    return results

def evaluate_fault_classification(y_true, y_pred, threshold=0.5):
    """
    Section V.B.3: Evaluation Metrics for Fault Type Classification (Dim 3).
    """
    y_pred_bin = (np.array(y_pred) >= threshold).astype(int)
    y_true = np.array(y_true)
    
    tp = np.sum((y_true == 1) & (y_pred_bin == 1), axis=0)
    fp = np.sum((y_true == 0) & (y_pred_bin == 1), axis=0)
    fn = np.sum((y_true == 1) & (y_pred_bin == 0), axis=0)
    
    # Macro metrics
    precision_macro = np.mean(np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)!=0))
    recall_macro = np.mean(np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)!=0))
    f1_macro = np.mean(np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2*tp+fp+fn)!=0))
    
    # Micro metrics
    tp_sum, fp_sum, fn_sum = np.sum(tp), np.sum(fp), np.sum(fn)
    precision_micro = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall_micro = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    f1_micro = (2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum) if (2 * tp_sum + fp_sum + fn_sum) > 0 else 0.0
    
    metrics = {
        'MiPr': precision_micro,
        'MaPr': precision_macro,
        'MiRe': recall_micro,
        'MaRe': recall_macro,
        'MiF1': f1_micro,
        'MaF1': f1_macro,
    }
    return metrics
