""" tracker3d/metrics.py

This file contains functions useful for measuring how well a keras model
performs at predicting results for the tracking problem.

Author: Daniel Zurawski
Author: Keshav Kapoor
Organization: Fermilab
Grammar: Python 3.6.1
"""

from .utils import to_categorical, from_categorical
from .utils import plot3d, display_side_by_side
import numpy as np
import keras.backend as kb
from .tracker_types import PMatrix, Target, Train, Event
from typing import Any

# Keras tensors are different types depending on backend used.
# For instance, tensorflow uses a different type than theano.
Tensor = Any


def wrong_classifications(target: PMatrix,
                          prediction: PMatrix,
                          verbose: bool=False)\
        -> int:
    """ Get the number of incorrect classifications.

    Arguments:
        target (PMatrix):
            The ground truth probability matrix.
        prediction (PMatrix):
            The prediction probability matrix.
        verbose (bool):
            True if you want this function to print things.
            False if you want this function to keep its big mouth shut.

    Returns (int):
        The number of incorrect classifications.
    """
    wrong_category = 0  # The number of wrong classifications.

    # Iterate through the prediction rows and find wrong classifications.
    for j, row in enumerate(prediction):
        predict = np.argmax(row)  # The prediction for this row.
        answer  = np.argmax(target[j])
        if predict != answer:
            if verbose:
                print("The model predicted hit {} incorrectly."
                      .format(j))
                print("Certainty of prediction: {}."
                      .format(row[predict]))
                print("The correct track was {}."
                      .format(answer if answer < target.shape[1] else "noise"))
            wrong_category += 1
    if verbose:
        print("-- There were {} wrong classifications."
              .format(wrong_category))
    return wrong_category


def discretize(probs: PMatrix)\
        -> PMatrix:
    """ Return a discrete probability matrix.

    A discrete probability matrix is derived from a regular probability
    matrix. Each row in the discrete matrix contain a single 1 and the rest
    0's. The column that this 1 is placed in corresponds to the maximum
    probability within the regular probability matrix's row. In the case of a
    tie, choose the column with lowest index.

    probs:                return:
    [[0.3, 0.5, 0.2],     [[0, 1, 0],
     [0.2, 0.2, 0.6],  ->  [0, 0, 1],
     [0.4, 0.4, 0.2]]      [1, 0, 0]]

    Arguments:
        probs (PMatrix):
            A matrix consisting of positive entries and where each row
            adds up to exactly 1.

    Returns (PMatrix):
        Return a matrix of the same shape as probabilities such that each
        row contains a single 1 and the rest 0's.
    """
    return to_categorical(from_categorical(probs), probs.shape[1])


def hits_per_track(probs: PMatrix,
                   verbose: bool=False)\
        -> np.ndarray:
    """ Return an array of integers representing the number of hits per track.

    Arguments:
        probs (PMatrix):
            A probability matrix.
        verbose (bool):
            True if this functions should print anything. Else, no printing.

    Returns (np.ndarray):
        An array of integers. The cell at index *i* contains the number of
        hits that are a member of track *i* as defined by *probs*.
    """
    hits = discretize(probs).sum(axis=0)
    if verbose:
        for i, hit in enumerate(hits):
            print("Hits in Track {0}: {1}".format(i, hit))
    return hits


def probability_hits_per_track(targets: Target,
                               track_id: int,
                               num_hits: int,
                               verbose: bool=False)\
        -> float:
    """ Determine how well predictions are for a specific track id.

    Given a list of probability matrices *target*, a track id *track_id* and
    an expected number of hits per track *num_hits*, provide the percent of
    matrices that correctly assign this track id the correct number of hits.

    Arguments:
        targets (Target):
            A list of probability matrices.
        track_id (int):
            The id of the track.
        num_hits (int):
            The proper number of hits that a track is expected to have.
        verbose:
            If True, will print out the number of matrices that correctly
            assign the correct number of hits to the track with *track_id*.
            Else, prints nothing.

    Returns: (float)
    The percent of matrices that correctly assign *num_hits* number of hits to
    the track with track id, *track_id*.
    """
    answer = sum([(hits_per_track(m)[track_id] == num_hits) for m in targets])
    if verbose:
        print("Number correct: {}".format(answer))
    return answer / len(targets)


def discrete_accuracy(event: Event,
                      prediction: PMatrix,
                      target: PMatrix,
                      verbose: bool=False,
                      padding: bool=False)\
        ->float:
    """ Return the discrete accuracy for this prediction.

    The discrete accuracy is how measured as what percent of rows in
    *prediction* have the largest probability at the same index as the
    1 value within the target matrix's row.

    Arguments:
        event (Event):
            An array of hit coordinates.
        prediction (PMatrix):
            A matrix of track probability predictions for *train* event.
        target (PMatrix):
            A matrix of ground truth probabilities for *train* event.
        verbose (bool):
            True if graphs and tables should both be displayed.
            False if nothing will be displayed.
        padding (bool):
            True if padding should be removed.
            False if padding should be accounted for within this calculation.

    Returns: (float):
        The discrete accuracy
    """
    if padding:
        # Time to remove all the (0, 0, 0) hits and their corresponding rows
        # in *target* and *prediction*.
        zero = np.zeros(event.shape[1])
        pad_idx = []
        for i, hit in enumerate(event):
            if np.all(np.equal(hit, zero)):
                pad_idx.append(i)
        event      = np.delete(event,      pad_idx, axis=0)
        prediction = np.delete(prediction, pad_idx, axis=0)
        target     = np.delete(target,     pad_idx, axis=0)

    display_plots_and_tables = False  # If anything goes wrong, flips to True.
    discrete = discretize(prediction)

    # Time to calculate the accuracy
    accuracy = 0
    for i, row in enumerate(discrete):
        equivalent = np.equal(np.argmax(row), np.argmax(target[i]))
        accuracy  += int(equivalent)
        if verbose and not equivalent:
            print("Wrong hit in row: {}".format(i))
            display_plots_and_tables = True
    percent = accuracy / len(target)  # The percent correct.
    if verbose:
        print("The accuracy is: {}".format(percent))
        if display_plots_and_tables:
            plot3d(event, prediction, target)
            display_side_by_side(event, prediction, target)
    return percent


def discrete_accuracy_all(train: Train,
                          predictions: Target,
                          targets: Target,
                          padding: bool=True)\
        -> float:
    """ Return the average discrete accuracy.

    The discrete accuracy is how measured as what percent of rows in a
    prediction have the largest probability at the same index as the
    1 value within the target matrix's row.

    Arguments:
        train (Train):
            A list of training Events.
        predictions (Target):
            A list of predicted probability matrices.
        targets (Target):
            A list of ground truth probability matrices.
        padding (bool):
            True if padding should be removed.
            False if padding should be accounted for within this calculation.

    Returns: (float)
        Average discrete accuracy
    """
    zero     = np.zeros(train.shape[2])
    accuracy = 0
    count    = 0
    for i, event in enumerate(train):
        discrete = discretize(predictions[i])
        for j, hit in enumerate(event):
            if padding and np.all(np.equal(hit, zero)):
                # Skip padding events.
                continue
            else:
                accuracy += np.all(np.equal(discrete[j], targets[i][j]))
                count    += 1
    return accuracy / count


def print_discrete_metrics(train: Train,
                           predictions: Target,
                           targets: Target,
                           padding: bool=False)\
        -> None:
    """ Print some discrete matrix metrics.

    Arguments:
        predictions (PMatrix):
            A list of predictions.
        targets (PMatrix):
            A list of ground truths matrices.
        train (Train):
            A list of hit events.
        padding (bool):
            Tru

    Returns: (None)
    """
    for i, event in enumerate(predictions):
        print("The event number is: {}".format(i))
        discrete_accuracy(train[i], event, targets[i], verbose=True)
    acc = discrete_accuracy_all(train, predictions, targets, padding=padding)
    print("The overall accuracy is: {}".format(acc))


def categorical_accuracy_padding(y_true: Tensor,
                                 y_pred: Tensor)\
        -> Tensor:
    """ A custom metric to be compiled with a keras model.

    WARNING: DOES NOT WORK!!! :(

    Arguments:
        y_true (Tensor):
            The ground truth probability matrix as a tensor.
        y_pred (Tensor):
            The prediction probability matrix as a tensor.

    Returns (Tensor):
    """
    # First, remove all the padding rows from *y_true* and *y_pred*.
    true_array  = kb.get_value(y_true)  # The ground truth probability matrix.
    pred_array  = kb.get_value(y_pred)  # The prediction probability matrix.
    padding     = np.zeros(true_array.shape[1])  # A padding row.
    padding[-1] = 1.0  # Set the last element in the pad row to 1.
    pad_indices = np.where(np.all(true_array == padding, axis=1))
    true_array  = np.delete(true_array, pad_indices, axis=0)
    pred_array  = np.delete(pred_array, pad_indices, axis=0)
    true_const  = kb.constant(true_array, shape=true_array.shape)
    pred_const  = kb.constant(pred_array, shape=pred_array.shape)

    # Finally, return tensor indicating which rows were such that both *y_true*
    # and *y_pred* had maximum values at the same index in that particular row.
    return kb.cast(kb.equal(kb.argmax(true_const, axis=-1),
                            kb.argmax(pred_const, axis=-1)),
                   kb.floatx())
