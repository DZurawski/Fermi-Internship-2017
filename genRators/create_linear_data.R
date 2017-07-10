# create_linear_data.R
# Author: Daniel Zurawski
# Create a dataframe of linear tracks.
# Save a dataframe with columns (event_id, cluster_id, r, phi, z).

require("tidyr")
require("dplyr")

events.total <- 1000 # The total number of events to generate.
tracks.total <- 10 # The total number of tracks per event.
layers       <- c(1, 2, 3, 4, 5) # The layer radiuses.
z.bounds     <- c(-200, 200) # The minimum and maximum allowable z values.
hits.total   <- length(layers) * tracks.total # Total number of hits per event.
eta.bounds   <- c(atan(tail(layers, n=1) / z.bounds[1] - pi),
                  atan(tail(layers, n=1) / z.bounds[1]),
                  atan(tail(layers, n=1) / z.bounds[2]),
                  atan(tail(layers, n=1) / z.bounds[2] + pi))

stopifnot(eta.bounds[1] < eta.bounds[2]) # runif doesn't work correctly
stopifnot(eta.bounds[3] < eta.bounds[4]) # if these conditions are false.

hits <- data.frame(event_id=numeric(0), cluster_id=numeric(0),
                   phi=numeric(0), r=numeric(0), z=numeric(0))

for (event_id in seq(0, events.total - 1)) {
    for (cluster_id in seq(0, tracks.total - 1)) {
        phi <- runif(1, -pi, pi)
        eta <- sample(c(runif(1, eta.bounds[1], eta.bounds[2]),
                        runif(1, eta.bounds[3], eta.bounds[4])), 1)
        z     <- layers / tan(eta)
        track <- data.frame(event_id, cluster_id, phi, r=layers, z)
        hits  <- rbind(hits, track)
    }
}
hits <- hits %>% arrange(event_id, cluster_id, phi, r, z)

write.csv(hits, "../datasets/standard_linear.csv", row.names=TRUE)
