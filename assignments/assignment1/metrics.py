from sklearn.metrics import confusion_matrix
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    conf_matr = confusion_matrix(prediction, ground_truth)
    precision = conf_matr[1, 1] / (conf_matr[1, 1] + conf_matr[0, 1])
    recall = conf_matr[1, 1] / (conf_matr[1, 1] + conf_matr[1, 0])
    accuracy = (conf_matr[0, 0] + conf_matr[1, 1]) / len(prediction)
    f1 = 2 * (precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    counter = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            counter += 1
    acc = counter / len(prediction)
    return acc
