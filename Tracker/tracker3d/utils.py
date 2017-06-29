""" tracker3d/utils.py
This file contains utility functions for the tracker3d package.
@author: Daniel Zurawski
@organization: Fermilab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for '3d' projection
from IPython.display import display, HTML


def plot3D(hits, IDs):
    """ Display a 3D plot of hits.
    
    Arguments:
        hits (numpy.array):
            A 2D matrix such that each row contains a 3-element
            array (phi, r, z) that describe the position of a hit.
        IDs (list of integers): 
            A list of track IDs. The ID at index i determines the track ID of
            the hit in 'hits' at index i. IDs must be integers with minimum
            value of 0. The number of rows in hits must equal len(IDs).
    
    Returns:
        None
    """
    max_id = np.amax(IDs)
    XYZ    = np.apply_along_axis(_to_cartesian, 1, hits) # X Y Z coordinates.
    cmap   = plt.cm.get_cmap('hsv', max_id + 1) # Colors for each track.
    tracks = [[] for _ in range(max_id + 1)]
    for i, ID in enumerate(IDs):
        tracks[ID].append(XYZ[i])
    tracks = [np.array(track) for track in tracks]
    
    plot = plt.figure().add_subplot(111, projection='3d')    
    plot.scatter(xs=tracks[max_id][:,0],
                 ys=tracks[max_id][:,1],
                 zs=tracks[max_id][:,2],
                 c=cmap(max_id),
                 marker='x')
    for i in range(max_id):
        plot.plot(xs=tracks[i][:,0],
                  ys=tracks[i][:,1],
                  zs=tracks[i][:,2],
                  c=cmap(i),
                  linestyle='-',
                  marker='o')
    plt.show(plot)
### END FUNCTION create_plot

def graph_losses(histories):
    """ Graph the accuracy and loss of a model's histories.
    
    This function graphs neural network model loss.
    This is code from HEPTrks keras tutorial file, in DSHEP folder. 
    
    Arguments:
        histories (list):
            A list of pairs such that pair.first is the label for this history
            and pair.second is the fitting history for a keras model.
            
    Returns:
        None
    """
    plt.figure(figsize=(10,10))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    colors=[]
    do_acc=False
    for label,loss in histories:
        color = tuple([0.1, 0.1, 0.1])
        colors.append(color)
        l = label
        vl= label+" validation"
        if 'acc' in loss.history:
            l+=' (acc %2.4f)'% (loss.history['acc'][-1])
            do_acc = True
        if 'val_acc' in loss.history:
            vl+=' (acc %2.4f)'% (loss.history['val_acc'][-1])
            do_acc = True
        plt.plot(loss.history['loss'], label=l, color=color)
        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=2, ls='dashed',
                     label=vl, color=color)
    plt.legend()
    plt.yscale('log')
    plt.show()
    if not do_acc: return
    plt.figure(figsize=(10,10))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    for _,(label,loss) in enumerate(histories):
        color = tuple([0.0, 0.0, 1.0])
        if 'acc' in loss.history:
            plt.plot(loss.history['acc'], lw=2, label=label+" accuracy",
                     color=color)
        if 'val_acc' in loss.history:
            plt.plot(loss.history['val_acc'], lw=2, ls='dashed',
                     label=label+" validation accuracy", color=color)
    plt.legend(loc='lower right')
    plt.show()
### END FUNCTION show_losses

def from_categorical(matrix):
    """ An inverse function to keras.utils.to_categorical().
    
    Arguments:
        matrix (2D numpy array):
            A probability matrix
    
    Returns:
        An array of indices at which each row reaches its maximum value.
    """
    return np.argmax(matrix, axis=1)
# END FUNCTION from_categorical

def multi_column_df_display(list_dfs, cols=2):
    """ Displays a list of pd.DataFrames in IPython as a table with 'cols'
    number of columns.
    
    Code by David Medenjak responding to StackOverflow question found here:
    https://stackoverflow.com/questions/38783027/jupyter-notebook-display-
    two-pandas-tables-side-by-side
    
    Arguments:
        list_dfs (list):
            A list of pd.DataFrames to display.
    
    Returns:
        None
    """
    html_table = "<table style='width:100%; border:0px'>{content}</table>"
    html_row   = "<tr style='border:0px'>{content}</tr>"
    html_cell  = ("<td style='width:{width}%;vertical-align:top;border:0px'>"
                  "{{content}}</td>")
    html_cell  = html_cell.format(width=100/cols)

    cells = [ html_cell.format(content=df.to_html()) for df in list_dfs ]
    cells += (cols - (len(list_dfs)%cols))*[html_cell.format(content="")] # pad
    rows = [ html_row.format(content="".join(cells[i:i+cols]))
            for i in range(0,len(cells),cols)]
    display(HTML(html_table.format(content="".join(rows))))
# END FUNCTION multi_column_df_display

def display_side_by_side(train, target, maximum=2):
    """ Get a feel for how the data looks by printing out the first
    'maximum' events in 'train' and 'target'.
    
    Arguments:
        train (np.array):
            The training data.
        target (np.array):
            The target data.
        maximum (np.array):
            The maximum number of events to print out. The events to print out
            are events with indices [0, maximum).
    
    Returns:
        None
    """
    print("Displaying the first {} inputs and outputs.".format(maximum))
    print("Input shape:  {}".format(train.shape))
    print("Output shape: {}".format(target.shape))

    train_cols  = ["phi", "layer", "z"]
    target_cols = ["T{}".format(i) for i in range(target.shape[2] - 1)] + ["N"]

    input_frames  = [pd.DataFrame(data=train[i],
                                  columns=train_cols) for i in range(maximum)]
    output_frames = [pd.DataFrame(data=target[i],
                                  columns=target_cols) for i in range(maximum)]    
    df_list  = []
    for i in range(len(input_frames)):    
        df_list.append(input_frames[i])
        df_list.append(output_frames[i])
    
    multi_column_df_display(df_list)
# END FUNCTION display_side_by_side

def print_scores(model, train, target, batch_size):
    """ Print out evaluation score and accuracy from a model.
    
    Arguments:
        model (keras model):
            The keras model to evaluate.
        train (np.array):
            The input data that the model trained on.
        target (np.array):
            The output data that the model trained on.
        batch_size (int):
            The batch size for evaluation.
        
        Returns:
            None
    """
    score, acc = model.evaluate(train, target, batch_size=batch_size)
    print("\nTest Score:    {}".format(score))
    print("Test Accuracy: {}".format(acc))
# END FUNCTION print_scores

def print_metrics(train, target, predictions):
    for i, prediction in enumerate(predictions):
        print("Event {}".format(i))
        false_negative = 0
        false_positive = 0
        wrong_category = 0
        for j, row in enumerate(prediction):
            predict = np.argmax(row)
            answer  = np.argmax(target[i,j])
            if predict != answer:
                print("\tThe model predicted event {0}, hit {1} incorrectly,"
                      .format(i, j))
                print("\t\twith {0} accuracy. The correct track was {1}"
                      .format(row[predict],
                              answer if answer < target.shape[2] else "noise"))
                if predict == target.shape[2] - 1:
                    print("\tPredicted as noise when answer was not noise.")
                    false_negative += 1
                elif answer == target.shape[2] - 1:
                    print("\tPredicted as track when answer was noise.")
                    false_positive += 1
                else:
                    print("\tPredicted as track {0} when answer was track {1}"
                          .format(predict, answer))
                    wrong_category += 1
        print("=== There were {} false negatives (predicted as noise, but track"
              .format(false_negative))
        print("=== There were {} false positives (predicted as track, but noise"
              .format(false_positive))
        print("=== There were {} wrong classifications."
              .format(wrong_category))
# END FUNCTION print_metrics

def _to_cartesian(hit):
    """ Transform the hit (phi, r, z) tuple into cartesian coordinates. """
    phi, r, z = hit[0], hit[1], hit[2]
    return (np.cos(phi) * r, np.sin(phi) * r, z)
### END function _to_cartesian

if __name__ == "__main__":
    import tracker3d.loader
    frame = pd.read_csv("../datasets/standard_linear.csv")
    train, target = tracker3d.loader.dataload(frame, nev=10, tpe=8, ts=4, npe=50)
    idx = np.random.randint(0, 10)
    print("Event {}".format(idx))
    IDs = tracker3d.utils.from_categorical(target[idx])
    #tracker3d.utils.plot3D(train[idx], IDs)