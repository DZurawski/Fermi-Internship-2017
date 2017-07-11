=================
=== tracker3d ===
=================

Author: Daniel Zurawski
Author: Keshav Kapoor
Organization: Fermilab

tracker3d -- A package containing modules used to create neural network model
            training data and to measure how well these models do.

loader.py -- A module for loading in data to a train/target format. This module
                also contains data saving and loading from file functions.

utils.py -- A module for utilities such as graphing and displaying results
                from a neural network.

metrics.py -- A module for measuring how well a neural network model does.

tracker_types.py -- A small module for defining types. These types are used to
                provide type hints for functions.