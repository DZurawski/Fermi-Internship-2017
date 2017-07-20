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


def plot3d(hits: Event,
           prediction: PMatrix,
           target: Optional[PMatrix]=None,
           order: Tuple[str]=("phi", "r", "z"),
           title: str="",
           padding: bool=True,
           has_noise: bool=True,
           flat_ax: Optional[str]=None)\
        -> Any:
    """ Display a 3D plot of hits.

    If flat_ax is None (indicating to plot in 3d), then you can perform these
    actions on the plot.
        1. Click on a hit to display information about it.
        2. Click on a track node to highlight that track.
        3. Press the spacebar to toggle layer cylinders.

    These actions have not been implemented for the 2d plots.

    Arguments:
        hits (Sequence[Hit]):
            A list of hits to display as points within the 3d plot.
        prediction (PMatrix):
            A probability matrix describing the track predictions of each hit.
        target: (PMatrix):
            The actual probability matrix that describes the true tracks that
            each hit belongs to.
        order: (Tuple[str]):
            The ordering of phi, r, z for a hit.
        title: (str):
            The title of this plot.
        padding: (bool)
            If True, then don't show the last track column. If padding was
            included in the event, the last track column is the padding column.
        flat_ax: Optional[str])
            The axis to flatten if you want to plot 2 dimensions. None if 3d.
        has_noise (bool):
            True if there is a noise column in the PMatrices.

    Returns: Plot3D
        WARNING!!!!
        Please assign this to some variable if you are using this function
        in jupyter notebook. Otherwise, it gets garbage-collected if you plot
        more than one time.
    """
    if target is None:
        plot = Plot3D(
                hits,
                prediction,
                prediction,
                order,
                flat_ax,
                padding,
                has_noise
        )
    else:
        plot = Plot3D(
                hits,
                prediction,
                target,
                order,
                flat_ax,
                padding,
                has_noise
        )
    plot.plot(title)
    return plot  # Please assign this to something after function.
    

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

    # Create pandas frames from the input data.
    input_frames  = pd.DataFrame(data=train, columns=order)
    output_frames = (pd.DataFrame(data=target.round(2), columns=target_cols)
                     .replace(0, ""))

    df_list = [input_frames, output_frames]

    # If predictions was specified, add it to the data frame list *df_list*.
    if predictions is not None:
        print("Prediction shape: {}".format(predictions.shape))
        prediction_frames = (pd.DataFrame(data=predictions.round(2),
                                          columns=target_cols)
                             .replace(0.00, ""))
        df_list.append(prediction_frames)

    # Display the data frame list using HTML.
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


class Plot3D:
    """ A 3D plot displaying an event. """
    def __init__(self,
                 event: Event,
                 pred: PMatrix,
                 target: PMatrix,
                 order: Tuple[str] = ("phi", "r", "z"),
                 flat_ax: Optional[str]=None,
                 padding: bool=True,
                 has_noise: bool=True)\
            -> None:
        """ Initialize a Plot3D.

        Arguments:
            event (Event):
                An event to display.
            pred (PMatrix):
                A prediction probability matrix.
            target (PMatrix):
                A target probability matrix.
            order (Tuple[str]):
                The order that hits in an event are in.
            padding (bool):
                True if plot should not show last column (pad column).
            has_noise (bool):
                If this target matrix has a noise column, then True.

        Returns: (None)
        """
        self.display_text = None

        self.ids     = from_categorical(pred)  # List of cluster ids.
        self.act_ids = from_categorical(target)  # List of actual ids.
        self.probs   = from_categorical_prob(pred)  # List of probabilities.
        self.padding = padding  # True if plot should ignore last track.
        self.cylinders = []  # Storage area for layer cylinder displays.

        self.num_ids = len(self.ids)  # Number of ids.
        self.tracks  = [[] for _ in range(self.num_ids)]  # List of tracks.
        self.radius  = np.unique(event[:, order.index("r")])  # Unique radiuses.
        self.pos2idx = dict()  # Position to index dictionary.
        self.idx2col = dict()  # Index to collection dictionary
        self.col2col = dict()  # Legend PathCollection to Plot PathCollection.

        self.flat_ax = flat_ax  # Which axis to flatten in 2d display.
        self.fig     = plt.figure()  # Plot figure.
        self.leg     = None  # Plot legend.
        self.collection = None  # Current collection that is selected by user.
        self.has_noise  = has_noise

        # If flat_ax is not None, then the plot will be 2-dimensional.
        if self.flat_ax is None:
            self.ax = Axes3D(self.fig)
        else:
            self.ax = plt.subplot(111)

        # Extract the cartesian coordinates from the input cylindrical ones.
        self.hits = np.apply_along_axis(
                lambda x: self._to_cartesian(x, order=order),
                1, event)

        # Assign hits to tracks and positions to *self.hits* indices.
        for idx, ID in enumerate(self.ids):
            self.tracks[ID].append(self.hits[idx])
            self.pos2idx[tuple(self.hits[idx])] = idx

        # Grab a mapping from track id to actual track length.
        unique, counts = np.unique(self.act_ids, return_counts=True)
        self.id2count = dict(zip(unique, counts))
        self.func_reference = []  # Reference to event functions

        self.cylinder_toggle_button = " "

    def plot(self, title: str=None)\
            -> None:
        """ Plot this 3D plot.

        Arguments:
            title: (str)
                The title of this plot.

        Returns: (None)
        """
        # Add the tracks to the plot.
        legend_index = 0  # The index for a legend handle.
        zeros = np.zeros(3)
        for i in range(self.num_ids):
            if not self.tracks[i]:
                # Don't display empty tracks.
                continue
            track = np.array(self.tracks[i])
            if self.padding and np.all(np.equal(track[0], zeros)):
                # Don't display padding sequences.
                continue
            else:
                # We are all fine. Let's plot this track.
                if self.flat_ax is None:
                    self._plot3d(track, legend_index)
                else:
                    self._plot2d(track, legend_index)
                legend_index += 1

        # Add a legend that can be clicked.
        self.leg = self.ax.legend(loc='upper right', fancybox=True, shadow=True)
        for i, handle in enumerate(self.leg.legendHandles):
            handle.set_picker(5)
            self.col2col[handle] = self.idx2col[i]

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        if self.flat_ax is None:
            self.ax.set_zlabel("Z")
            self.func_reference.append(self._on_pick)  # So gc does not eat it.
            self.func_reference.append(self._toggle_cylinders)
            self.fig.canvas.mpl_connect('pick_event',
                                        self.func_reference[0])
            self.fig.canvas.mpl_connect('key_press_event',
                                        self.func_reference[1])
            self.ax.set_xlim3d(-1000, 1000)
            self.ax.set_ylim3d(-1000, 1000)
            self.ax.set_zlim3d(-200, 200)
        elif self.flat_ax == "z":
            for r in self.radius:
                self.ax.add_artist(
                        plt.Circle(
                                (0, 0),
                                r,
                                color='black',
                                fill=False,
                                linestyle='-',
                                alpha=0.1))

        if self.has_noise:
            self.leg.get_texts()[-1].set_text("Noise")
        self.ax.set_title(title)
        plt.show(self.ax)

    def _on_pick(self, event)\
            -> None:
        """ Either display hit info or highlight a track after a pick event.

        Arguments:
            event:
                A pick event.

        Returns: (None)
        """
        new_collection = self.col2col.get(event.artist)
        if new_collection is None:
            # Looks like we picked a plot point.
            self._on_pick_hit(event)
        else:
            # Looks like we picked a legend point.
            self._on_pick_legend(event, new_collection)
        self.fig.canvas.draw()

    def _on_pick_legend(self, event, new_collection)\
            -> None:
        """ Highlight the track associated with this pick.

        Arguments:
            event:
                A pick event.
            new_collection: (Path3DCollection)
                The new collection of points to highlight.

        Returns: (None)
        """
        if self.collection is not None:
            # Make previous collection go back to un-highlighted.
            self.collection._edgecolor3d = np.array([[0, 0, 0, 1]])
            self.collection.set_linewidth(1)
        if self.collection != new_collection:
            # If there is new collection, highlight it.
            self.collection = self.col2col[event.artist]
            pos     = np.array(self.collection._offsets3d)
            index   = self.pos2idx.get(tuple(pos[:3, 0]))
            t_count = self.id2count.get(self.act_ids[index])
            text    = "Track Length: {0}\nActual Length: {1}".format(
                            pos.shape[1],
                            t_count)
            self.collection._edgecolor3d = np.array([[0.8, 0.8, 0, .8]])
            self.collection.set_linewidth(10)
        else:
            # Just un-highlight the previous one. No new collection.
            text = ""
            self.collection = None

        if self.display_text is not None:
            self.display_text.remove()
        self.display_text = self.ax.text2D(0.1, 0.8, text,
                                           transform=self.ax.transAxes)

    def _on_pick_hit(self, event)\
            -> None:
        """ Display the hit information associated with this pick.

        Arguments:
            event:
                An event from a pick event.
        Returns: (None)
        """
        edx    = event.ind[0]
        artist = event.artist
        pos    = tuple(np.array(artist._offsets3d)[0:3, edx])
        hdx    = self.pos2idx[pos]
        text   = ("Hit: {0}, Track: {1}, Certainty: {2:.1f}%\nActual Track: {3}"
                  .format(hdx,
                          self.ids[hdx],
                          self.probs[hdx] * 100,
                          self.act_ids[hdx]))
        if self.display_text is not None:
            self.display_text.remove()
        self.display_text = self.ax.text2D(0.1, 0.8, text,
                                           transform=self.ax.transAxes)

    @staticmethod
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
        zippy = sorted(zip(order, hit), key=lambda pair: pair[0])
        sorty = [x for (y, x) in zippy]
        phi, r, z = sorty[0], sorty[1], sorty[2]
        return np.array([np.cos(phi) * r, np.sin(phi) * r, z])

    def _plot3d(self, track: np.ndarray, i: int)\
            -> None:
        """ Plot a 3d track. """
        self.idx2col[i] = self.ax.scatter3D(
                xs=track[:, 0], ys=track[:, 1], zs=track[:, 2],
                label="T{}".format(i),
                picker=True,
                s=50,
                linewidth=1,
                edgecolor='black',
                depthshade=False
        )

    def _plot2d(self, track: np.ndarray, i: int)\
            -> None:
        """ Plot a 2d track. """
        if self.flat_ax == "x":
            x, y = track[:, 1], track[:, 2]
        elif self.flat_ax == "y":
            x, y = track[:, 0], track[:, 2]
        else:
            x, y = track[:, 0], track[:, 1]
        self.idx2col[i] = self.ax.scatter(
                x=x, y=y,
                label="T{}".format(i),
                picker=True,
                s=50,
                linewidth=1,
                edgecolor='black',
        )

    def _plot_cylinder(self, r: float)\
            -> None:
        """ Plot a cylinder centered at (0, 0, 0) with radius *r*.

        The cylinder circles around the z axis.

        Arguments:
            r (float):
                The radius of the cylinder.

        Returns: (None)
        """
        xc, zc = np.meshgrid(
                np.linspace(-r, r, 25),
                np.linspace(-100, 100, 25))
        yc = np.sqrt(r**2 - xc ** 2)
        rstride = 20
        cstride = 20
        color = 'blue'
        self.cylinders.append(self.ax.plot_surface(
                xc,
                yc,
                zc,
                alpha=.1,
                color=color,
                rstride=rstride,
                cstride=cstride
        ))
        self.cylinders.append(self.ax.plot_surface(
                xc,
                -yc,
                zc,
                alpha=.1,
                color=color,
                rstride=rstride,
                cstride=cstride
        ))

    def _toggle_cylinders(self, event)\
            -> None:
        """ Toggle display of cylinders on or off.

        If the user presses spacebar, cylinders are toggled on or off. These
        cylinders represent the layers of the collider.

        Arguments:
            event:
                An event from matplotlib.

        Returns: (None)
        """
        if event.key == self.cylinder_toggle_button:
            if not self.cylinders:
                for r in self.radius:
                    self._plot_cylinder(r)
            else:
                for cylinder in self.cylinders:
                    cylinder.remove()
                self.cylinders = []
            self.fig.canvas.draw()
