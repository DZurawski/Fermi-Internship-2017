# add_z_to_100MeV.R
# Author: Daniel Zurawski
# Modify the RAMP data set found in "public_train_100MeV.csv" by adding a
# uniformly random linear-slope z component.
# Save a dataframe with columns (event_id, cluster_id, r, phi, z).

require("tidyr")
require("dplyr")

initial.frame <- read.csv("../datasets/public_train_100MeV.csv")
z.bounds   <- c(-200, 200) # What should the min and max z values be?
layer.max  <- max(initial.frame$layer) + 1
eta.bounds <- c(atan(layer.max / z.bounds[1] - pi),
                atan(layer.max / z.bounds[1]),
                atan(layer.max / z.bounds[2]),
                atan(layer.max / z.bounds[2] + pi))

stopifnot(eta.bounds[1] < eta.bounds[2]) # runif doesn't work correctly
stopifnot(eta.bounds[3] < eta.bounds[4]) # if these conditions are false.

write.csv(
    initial.frame %>%
        mutate(phi = atan2(y, x)) %>%
        group_by(event_id, cluster_id) %>%
        mutate(eta = sample(c(runif(1, eta.bounds[1], eta.bounds[2]),
                              runif(1, eta.bounds[3], eta.bounds[4])), 1)) %>%
        ungroup() %>%
        mutate(r = (layer + 1)) %>%
        mutate(z = (r / tan(eta))) %>%
        arrange(event_id, cluster_id, layer) %>%
        select(event_id, cluster_id, r, phi, z),
    "../datasets/standard_curves100MeV.csv",
    row.names = TRUE
)
