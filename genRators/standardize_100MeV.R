# standardize_100MeV.R
# Author: Daniel Zurawski
# Standardize the RAMP tracking data found in "public_train_100MeV.csv".
# Save a dataframe with columns (event_id, cluster_id, r, phi, z).

require("tidyr")
require("dplyr")

write.csv(
    read.csv("../datasets/public_train_100MeV.csv") %>%
        mutate(phi = atan2(y, x), z = c(0), r = (layer + 1)) %>%
        select(event_id, cluster_id, r, phi, z) %>%
        arrange(event_id, cluster_id, r, phi, z),
    "../datasets/standard_100MeV.csv",
    row.names = TRUE
)
