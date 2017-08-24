import numpy as np
import pandas as pd
import IPython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
from . import extractor as ext, utils, metrics


def display(
        frame   : pd.DataFrame,
        order   : List[str],
        guess   : Optional[np.ndarray] = None,
        mode    : str = "default",
        decimal : int = 2,
        ) -> None:
    """ Display the frame or guess through IPython.
    :param frame:
        A pd.DataFrame
    :param order:
        A permutation of ["phi", "r", "z"]
    :param guess:
        A prediction probability matrix.
    :param mode:
        One of ["default", "pairs"]
        If "pairs", then the answer is displayed in the same cell as the
        "guess" prediction. Format: "`ANSWER`[PREDICTION]"
    :param decimal:
        How many decimals places to round the guesses to.
    :return:
        None.
    """
    table  = pd.DataFrame(ext.extract_input(frame, order), columns=order)
    target = ext.extract_output(frame, order).round(0)
    column = [chr(65+i) for i in range(target.shape[1] - 2)] + ["noise", "pad"]
    if mode == "discrete pairs":
        guess = metrics.discrete(guess).round(0)
        data = []
        for x in range(len(guess)):
            row = []
            for y in range(len(guess[x])):
                if target[x, y] == 0 and guess[x, y] == 0:
                    row.append("")
                else:
                    t, g = int(target[x, y]), guess[x, y]
                    row.append("`{0}`[{1}]".format(t, g))
            data.append(row)
        out_table = pd.DataFrame(data=data, columns=column)
        table = pd.concat([table, out_table], axis=1)
    if mode == "pairs" and guess is not None:
        guess = guess.round(decimal)
        data   = []
        for x in range(len(guess)):
            row = []
            for y in range(len(guess[x])):
                if target[x, y] == 0 and guess[x, y] == 0:
                    row.append("")
                else:
                    t, g = int(target[x, y]), guess[x, y]
                    row.append("`{0}`[{1}]".format(t, g))
            data.append(row)
        out_table = pd.DataFrame(data=data, columns=column)
        table = pd.concat([table, out_table], axis=1)
    else:
        out_table = pd.DataFrame(data=target, columns=column).replace(0, "")
        table = pd.concat([table, out_table], axis=1)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.max_colwidth', 10000000):
        IPython.display.display(table)


def boxplot(
        data   : List[np.ndarray],
        title  : str  = "Box Plot",
        xlabel : str  = "X",
        ylabel : str  = "Y",
        fliers : bool = False,
        xticks : Optional[List] = None,
        ) -> None:
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.boxplot(data, showfliers=fliers)
    if xticks is not None:
        plt.xticks([i for i in range(len(data))], xticks)


class Plot2D:
    """ A plot of the data. """
    def __init__(
            self,
            frame : pd.DataFrame,
            order : List[str],
            guess : Optional[np.ndarray] = None,
            ) -> None:
        """ Initialize member variables for the plot. """
        guess    = ext.extract_output(frame, order) if guess is None else guess
        guess_id = utils.from_categorical(guess)
        frame    = frame.sort_values(order)
        frame    = frame.assign(guess_id=guess_id, index=frame.index)

        self.frame = utils.remove_padding(frame)
        self.order = order
        self.fig   = plt.figure()
        self.leg   = None
        self.ax    = plt.subplot(111)

    def plot(
            self,
            mode  : str,
            title : str = "",
            ) -> None:
        """ Plot this 3D plot. """
        mode   = mode.lower()
        tracks = utils.list_of_groups(self.frame, group="guess_id")
        for i, track in enumerate(tracks):
            guess_id = track.iloc[0]["guess_id"]
            label   = chr(65 + int(guess_id))
            extract = ext.extract_input(track, self.order)
            values  = self.get_values(extract, mode)
            self.ax.scatter(
                    x=values[0, :, 0],
                    y=values[0, :, 1],
                    label=label,
                    picker=True,
                    s=100,
                    linewidth=1,
                    edgecolor='black')
            for t in range(len(track)):
                self.ax.text(values[0, t, 0], values[0, t, 1],
                             label, size=8, zorder=10, color="white",
                             horizontalalignment="center",
                             verticalalignment="center")
        self.ax.set_title(title)
        if mode == "xy":
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            for r in pd.unique(self.frame["r"]):
                self.ax.add_artist(
                        plt.Circle((0, 0), r, color='black', fill=False,
                                   linestyle='-', alpha=0.1))
        elif mode == "yz":
            self.ax.set_xlabel("Y")
            self.ax.set_ylabel("Z")
        elif mode == "xz":
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Z")
        elif mode == "zr":
            self.ax.set_xlabel("Z")
            self.ax.set_ylabel("R")
            for r in pd.unique(self.frame["r"]):
                self.ax.plot([-200, 200], [r, r], alpha=0.1, color="black")
        self.leg = self.ax.legend(loc='upper right', fancybox=True)
        plt.show(self.ax)

    def get_values(
            self,
            values : np.ndarray,
            mode : str,
            ) -> np.ndarray:
        mode = mode.lower()
        ps   = values[:, self.order.index("phi")]
        zs   = values[:, self.order.index("z")]
        rs   = values[:, self.order.index("r")]
        xs   = np.cos(ps) * rs
        ys   = np.sin(ps) * rs
        if mode == "xy":
            return np.dstack((xs, ys))
        elif mode == "xz":
            return np.dstack((xs, zs))
        elif mode == "yz":
            return np.dstack((ys, zs))
        else:
            return np.dstack((zs, np.sqrt(xs ** 2 + ys ** 2)))


class Plot3D:
    """ A plot of the data. """
    def __init__(
            self,
            frame : pd.DataFrame,
            order : List[str],
            guess : Optional[np.ndarray] = None,
            ) -> None:
        """ Initialize member variables for the plot. """
        guess    = ext.extract_output(frame, order) if guess is None else guess
        guess_id = utils.from_categorical(guess)
        frame    = frame.sort_values(order)
        frame    = frame.assign(guess_id=guess_id, index=frame.index)

        self.frame = utils.remove_padding(frame)
        self.order = order
        self.fig   = plt.figure()
        self.leg   = None
        self.ax    = Axes3D(self.fig)

    def plot(
            self,
            title    : str = "",
            x_limits : Tuple[int, int] = (-1000, 1000),
            y_limits : Tuple[int, int] = (-1000, 1000),
            z_limits : Tuple[int, int] = (-200,   200),
            ) -> None:
        """ Plot this 3D plot. """
        tracks = utils.list_of_groups(self.frame, group="guess_id")
        for i, track in enumerate(tracks):
            guess_id = track.iloc[0]["guess_id"]
            label  = chr(65 + guess_id)
            values = self.cartesian(ext.extract_input(track, self.order))
            self.ax.scatter3D(
                    xs=values[0, :, 0],
                    ys=values[0, :, 2],
                    zs=values[0, :, 1],
                    label=label,
                    picker=True,
                    s=100,
                    linewidth=1,
                    edgecolor='black',
                    depthshade=True)
            for t in range(len(track)):
                self.ax.text(values[0, t, 0], values[0, t, 2], values[0, t, 1],
                             label, size=8, zorder=10, color="white",
                             horizontalalignment="center",
                             verticalalignment="center")
        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.set_zlabel("Y")
        self.ax.set_xlim3d(x_limits[0], x_limits[1])
        self.ax.set_ylim3d(z_limits[0], z_limits[1])
        self.ax.set_zlim3d(y_limits[0], y_limits[1])
        self.leg = self.ax.legend(loc='upper right', fancybox=True)
        plt.show(self.ax)

    def cartesian(
            self,
            values : np.ndarray,
            ) -> np.ndarray:
        """ Transform 'phi', 'z', 'r' coordinates to cartesian coordinates. """
        ps = values[:, self.order.index("phi")]
        zs = values[:, self.order.index("z")]
        rs = values[:, self.order.index("r")]
        xs = np.cos(ps) * rs
        ys = np.sin(ps) * rs
        return np.dstack((xs, ys, zs))
