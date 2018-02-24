import pandas as pd
from sklearn.metrics import confusion_matrix


def calc_final_score(cm):
    """
    Calculate final score from a confusion matrix. False negatives
    are penalized 5x as much as false positives::

        Score = ( TP + TN ) / ( TP + TN + FP + 5*FN )

    """
    if type(cm) == pd.DataFrame:
        score = ((cm.iloc[1, 1] + cm.iloc[0, 0])
                  / (cm.iloc[1, 1] + cm.iloc[0, 0] + cm.iloc[0, 1] + 5*cm.iloc[1, 0]))
    elif type(cm) == np.ndarray:
        score = (cm[0, 0] + cm[0, 1]) / (cm[1, 1]+cm[0, 1]+cm[0, 1] + 5*cm[1, 0])

    return score


def calc_results(y_true, y_pred):
    """
    Calculate performance metrics. Input variables are array-likes of true
    outcomes and predicted outcomes.

    Returns the confusion matrix, the proportion of correct predictions,
    and the final score
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    cm = pd.DataFrame(cm, columns=['Predict 0', 'Predict 1'], index=['Actual 0', 'Actual 1'])

    # Correct classification proportion
    p_correct = (cm.iloc[0,0]+cm.iloc[1,1])/len(y_pred)

    # Score = ( TP + TN ) / ( TP + TN + FP + 5*FN )
    score = calc_final_score(cm)

    return cm, p_correct, score


def print_results(cm, pcorrect, score, classifier_name=''):
    """
    Display the performance results

    """
    print('Classifier: %s' % classifier_name)
    print('Confusion Matrix:')
    display(cm)
    print('Proportion Correct:', pcorrect)
    print('Final Score:', score)
    print('\n\n')
