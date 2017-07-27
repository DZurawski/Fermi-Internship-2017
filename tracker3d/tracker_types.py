""" tracker3d/tracker_types.py

This file contains type descriptions for type hints.
This is purely for organizational purposes, so that function typing
within other modules is easy to discern.

Author: Daniel Zurawski
Author: Keshav Kapoor
Organization: Fermilab
Grammar: Python 3.6.1
"""

# Currently, the numpy and typing modules do not support specific numpy type
# declarations (such as specifying the shape of an array). This is why many
# of the below user types have the same numpy type (np.ndarray). They should
# still be treated as distinct types for notation and typing purposes.

import numpy as np

# A 1D array with columns representing position data of a hit within an event.
Hit = np.ndarray

# A 2D array with rows representing Hits. The Hits within a Track all must
# be a member of the same cluster/track.
Track = np.ndarray

# A 2D array with rows representing Hits. The Hits within an Event do not
# necessarily have to be members of the same cluster/track.
Event = np.ndarray

# A 2D array with columns representing Tracks, rows representing Hits and
# individual cells representing the probability that this row's Hit
# is a member of this column's Track.
PMatrix = np.ndarray

# A 3D array where each depth is an Event. This is a collection of Event
# data that is used to train a model.
Train = np.ndarray

# A 3D array where each depth is a PMatrix. This is a collection of PMatrix
# data that is used to train a model. The Probs matrix at index *i* corresponds
# to the Hit within Train at index *i*.
Target = np.ndarray
