""" tracker3d/utils.py
This file contains utility functions for the tracker3d package.
@author: Daniel Zurawski
@organization: Fermilab
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    for i,(label,loss) in enumerate(histories):
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

def _to_cartesian(hit):
    """ Transform the hit (phi, r, z) tuple into cartesian coordinates. """
    phi, r, z = hit[0], hit[1], hit[2]
    return (np.cos(phi) * r, np.sin(phi) * r, z)
### END function _to_cartesian