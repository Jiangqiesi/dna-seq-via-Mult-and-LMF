import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", accuracy_score(binary_truth, binary_preds))

    print("-" * 50)


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()
        
        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
            test_truth_i = test_truth[:,emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()
        
        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds,axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)


def eval_dna(results, truths):
    index_to_base = ['A', 'C', 'G', 'T']
    base_arr = np.array([1, 1, 1, 1])
    results = results.numpy()
    truths = truths.numpy()
    results_indices = np.argmax(results, axis=-1)
    truths_indices = np.argmax(truths, axis=-1)
    print("results after argmax:", results_indices)
    print("truths after argmax:", truths_indices)

    list_of_results = []
    list_of_truths = []
    list_of_confidence = []
    list_of_mean = []
    for i in range(0, results_indices.shape[0]):
        result_str = ""
        truth_str = ""

        # confidence_arr = np.zeros(260)
        # 比较预测和真实标签
        # correct_predictions = (results_indices[i] == truths_indices[i])
        # 计算置信度
        # confidence_scores = results[np.arange(results.shape[1]), results_indices[i]]
        # mean_confidence = np.mean(confidence_scores[correct_predictions])
        for j in range(0, results_indices.shape[1]):
            result_str = result_str + index_to_base[results_indices[i][j]]
            truth_str = truth_str + index_to_base[truths_indices[i][j]]

        list_of_results.append(result_str)
        list_of_truths.append(truth_str)
        # list_of_confidence.append(confidence_scores)
        # list_of_mean.append(mean_confidence)

    correct_predictions = (results_indices == truths_indices)
    confidence_scores = np.max(results, axis=-1)
    mean_confidence = np.mean(confidence_scores[correct_predictions], axis=-1)
    print("shape of mean:", mean_confidence.shape)
    confidence_scores = np.abs(confidence_scores)
    # print("shape of confidence:", list_of_confidence[0].shape)

    with open("./final_data.txt", "w") as f:
        for i in range(0, len(list_of_results)):
            # 先输出索引
            f.write(str(i) + "\n")
            f.write("result:" + list_of_results[i] + "\n")
            f.write("truth:" + list_of_truths[i] + "\n")
            f.write("Confidence scores:" + str(confidence_scores[i]) + "\n")
            # f.write("Mean confidence:" + str(mean_confidence[i]) + "\n")

