import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0

    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0

    if precision + recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """
    pass


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    y_true_mean = sum(y_true) / len(y_true)
    
    ss_total = sum((y_true - y_true_mean) ** 2)
    
    ss_res = sum((y_true - y_pred) ** 2)
    
    r2 = 1 - (ss_res / ss_total)
    
    return r2
    


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    sq_diff = (y_true - y_pred) ** 2
    
    mse = sum(sq_diff) / len(y_true)
    
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    abs_diff = abs(y_true - y_pred)
    mae = sum(abs_diff) / len(y_true)
    
    return mae
    