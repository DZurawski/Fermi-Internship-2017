""" tracker3d/utils.py

This file contains utility functions for the tracker3d package.

Author: Daniel Zurawski
Author: Keshav Kapoor
Organization: Fermilab
Grammar: Python 3.6.1
"""

import numpy as np
import pandas as pd
import IPython.display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for '3d' projection
from typing import Optional, Sequence, Tuple, Any
from .tracker_types import Event, PMatrix, Hit, Train, Target

_hit_info_display_text = None  # Text to be used to display info on a plot.


def plot3d(hits: Event,
           probs: PMatrix,
           actual: Optional[PMatrix]=None,
           order: Tuple[str]=("phi", "r", "z"))\
        -> None:
    """ Display a 3D plot of hits.

    Arguments:
        hits (Sequence[Hit]):
            A list of hits to display as points within the 3d plot.
        probs (PMatrix):
            A probability matrix describing the track predictions of each hit.
        actual: (PMatrix):
            The actual probability matrix that describes the true tracks that
            each hit belongs to.
        order: (Tuple[str])
            The ordering of phi, r, z for a hit.

    Returns: (None)
    """
    maxsize = 50  # The maximum size a point can take.
    minsize = 10  # The minimum size a point can take.

    # Set up the information regarding track identification.
    ids   = from_categorical(probs)
    probs = from_categorical_prob(probs)
    if actual is not None:
        act_ids = from_categorical(actual)  # Actual ids

    # Transform the hits into cartesian coordinates.
    xyz = np.apply_along_axis(lambda x: _to_cartesian(x, order=order), 1, hits)

    num_ids = len(ids)

    # Set up a way to identify the tracks for each hit
    tracks = [[] for _ in range(num_ids)]
    sizes  = [[] for _ in range(num_ids)]
    for i, ID in enumerate(ids):
        tracks[ID].append(xyz[i])
        sizes[ID].append((maxsize - minsize) * probs[i] + minsize)

    # Time to plot the data.
    fig = plt.figure()
    ax  = Axes3D(fig)

    global _hit_info_display_text  # Used to display text on the plot.
    _hit_info_display_text = None

    # Define a function to display info on a hit if the user clicks on it.
    def _on_pick_hit(event) -> None:
        global _hit_info_display_text
        edx    = event.ind
        artist = event.artist
        pos    = (np.array(artist._offsets3d)[:, edx]).flatten()
        hdx    = np.where(xyz == pos)[0][0]
        text   = ("Hit {0} :: Track {1} :: Certainty {2}"
                  .format(hdx, ids[hdx], probs[hdx]))
        if actual is not None:
            text += ("\nActual Track {0}".format(act_ids[hdx]))
        if _hit_info_display_text is not None:
            _hit_info_display_text.remove()
        _hit_info_display_text = ax.text2D(0.05, 0.9, text,
                                           transform=ax.transAxes)
        fig.canvas.draw()

    # Plot each track. Empty tracks are not displayed.
    for i in range(num_ids):
        if not tracks[i]:
            continue
        track = np.array(tracks[i])
        if np.all(np.equal(track[0], np.array([0, 0, 0]))):
            continue  # Don't display padding sequences.
        ax.scatter(xs=track[:, 0],
                   ys=track[:, 1],
                   zs=track[:, 2],
                   label="T{}".format(i),
                   s=sizes[i],
                   picker=True,
                   depthshade=False)
    fig.canvas.mpl_connect('pick_event', _on_pick_hit)
    plt.legend()
    plt.show(ax)
    

def graph_losses(histories: Sequence[Tuple[str, Any]])\
        -> None:
    """ Graph the accuracy and loss of a model's histories.
    
    This function graphs neural network model loss.
    This is code from HEPTrks keras tutorial file, in DSHEP folder. 
    
    Arguments:
        histories (List[Tuple[str, History]]):
            A list of pairs such that pair.first is the label for this history
            and pair.second is the fitting history for a keras model.
            
    Returns: (None)
    """
    plt.figure(figsize=(10, 10))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    colors = []
    do_acc = False
    for label, loss in histories:
        color = tuple([0.1, 0.1, 0.1])
        colors.append(color)
        l  = label
        vl = label+" validation"
        if 'acc' in loss.history:
            l += ' (acc %2.4f)' % (loss.history['acc'][-1])
            do_acc = True
        if 'val_acc' in loss.history:
            vl += ' (acc %2.4f)' % (loss.history['val_acc'][-1])
            do_acc = True
        plt.plot(loss.history['loss'], label=l, color=color)
        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=2, ls='dashed',
                     label=vl, color=color)
    plt.legend()
    plt.yscale('log')
    plt.show()
    if not do_acc:
        return
    plt.figure(figsize=(10, 10))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    for _, (label, loss) in enumerate(histories):
        color = tuple([0.0, 0.0, 1.0])
        if 'acc' in loss.history:
            plt.plot(loss.history['acc'], lw=2, label=label+" accuracy",
                     color=color)
        if 'val_acc' in loss.history:
            plt.plot(loss.history['val_acc'], lw=2, ls='dashed',
                     label=label+" validation accuracy", color=color)
    plt.legend(loc='lower right')
    plt.show()


def to_categorical(ids: Sequence[int],
                   columns: int)\
        -> PMatrix:
    """ Create a probability matrix from a list of ids.
    
    Arguments:
        ids (Sequence[int]):
            A list of track IDs
        columns (int):
            The number of columns
    
    Returns: (PMatrix)
        numpy 2D array probability matrix
    """
    matrix = np.zeros((len(ids), columns))
    for i, ID in enumerate(ids):
        matrix[i][ID] = 1
    return matrix


def from_categorical(matrix: PMatrix)\
        -> np.ndarray:
    """ An inverse function to tracker3d.utils.to_categorical().

    Take a PMatrix matrix and turn it into a categorical list.
    
    Arguments:
        matrix (PMatrix):
            A probability matrix
    
    Returns: (np.ndarray)
        An array of indices at which each row reaches its maximum value.
    """
    return np.argmax(matrix, axis=1)


def from_categorical_prob(matrix: PMatrix)\
        -> np.ndarray:
    """ Return an array of maximum probabilities for each row in matrix.
    
    Arguments:
        matrix (PMatrix):
            A probability matrix
    
    Returns: (np.ndarray)
        An array of maximum probabilities from each row.
    """
    return np.amax(matrix, axis=1)


def multi_column_df_display(list_dfs: Sequence[pd.DataFrame],
                            cols: int=2)\
        -> None:
    """ Displays a list of pd.DataFrames in IPython as a table with 'cols'
    number of columns.
    
    Code by David Medenjak responding to StackOverflow question found here:
    https://stackoverflow.com/questions/38783027/jupyter-notebook-display-
    two-pandas-tables-side-by-side

    Code has been edited to apply better to this module.
    
    Arguments:
        list_dfs (list):
            A list of pd.DataFrames to display.
        cols (int):
            The number of columns the table should have.
    
    Returns: (None)
    """
    html_table = "<table style='border:10px'>{content}</table>"
    html_row   = "<tr style='border:10px'>{content}</tr>"
    html_cell  = "<td style='border:10px'>{content}</td>"
    cells      = [html_cell.format(content=df.to_html()) for df in list_dfs]
    rows       = [html_row.format(content="".join(cells[i:i+cols]))
                  for i in range(0, len(cells), cols)]
    ipydisplay = IPython.display
    all_html   = ipydisplay.HTML(html_table.format(content="".join(rows)))
    ipydisplay.display(all_html)


def display_side_by_side(train: Event,
                         target: PMatrix,
                         predictions: Optional[PMatrix]=None,
                         order: Sequence[str]=("phi", "r", "z"))\
        -> None:
    """ Display an event of train, target and predictions data on IPython.
    
    Arguments:
        train (Event):
            The training data to display.
        target (PMatrix):
            The target data to display.
        predictions (Optional[PMatrix]):
            A PMatrix matrix of predicted probabilities.
        order (Sequence[str]):
            An ordering for the Hit frame's labels.
    
    Returns: (None)
    """
    order = list(order)
    target_cols = ["T{}".format(i) for i in range(target.shape[1])]

    input_frames  = pd.DataFrame(data=train, columns=order)
    output_frames = (pd.DataFrame(data=target.round(2), columns=target_cols)
                     .replace(0, ""))

    df_list = [input_frames, output_frames]
    
    if predictions is not None:
        print("Prediction shape: {}".format(predictions.shape))
        prediction_frames = (pd.DataFrame(data=predictions.round(2),
                                          columns=target_cols)
                             .replace(0.00, ""))
        df_list.append(prediction_frames)
        
    multi_column_df_display(df_list, len(df_list))


def print_scores(model: Any,
                 train: Train,
                 target: Target,
                 batch_size: int)\
        -> None:
    """ Print out evaluation score and accuracy from a model.
    
    Arguments:
        model (keras model):
            The keras model to evaluate.
        train (Train):
            The input data that the model trained on.
        target (Target):
            The output data that the model trained on.
        batch_size (int):
            The batch size for evaluation.
        
    Returns: (None)
    """
    score, acc = model.evaluate(train, target, batch_size=batch_size)
    print("\nTest Score:    {}".format(score))
    print("Test Accuracy: {}".format(acc))


def _to_cartesian(hit: Hit,
                  order: Sequence[str]=("phi", "r", "z"))\
        -> Hit:
    """ Transform the hit tuple into cartesian coordinates.

    Arguments:
        hit (Hit):
            A Hit with phi, r, z coordinates.
        order (Sequence[str]):
            A sequence of strings that define how the Hit's phi, r, z
            coordinates are ordered.
            Example: order=("r", "phi", "z")
                     order=("z", "r", "phi")

    Returns: (Hit)
        A Hit, but with its position transformed into cartesian (XYZ)
        coordinates.
    """
    order  = list(order)
    sortie = [x for (y, x) in sorted(zip(order, hit),
                                     key=lambda pair: pair[0])]
    phi, r, z = sortie[0], sortie[1], sortie[2]
    return np.array([np.cos(phi) * r, np.sin(phi) * r, z])
