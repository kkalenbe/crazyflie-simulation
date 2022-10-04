"""synthetic_data_processing script."""

import numpy as np
import cv2
import random
random.seed()

def process_camera_image(camera_image_raw):
    camera_image_processed = np.copy(camera_image_raw)

    # Convert to grayscale
    camera_image_processed = cv2.cvtColor(camera_image_processed, cv2.COLOR_RGB2GRAY)

    # Possibly crop to reduced size

    # Add overexposure artifacts

    return camera_image_processed

def process_tof_image(tof_image_raw):
    tof_image_processed = np.copy(tof_image_raw)

    min_distance = 0.2                  # Min sensor measurement range
    max_distance = 3.0                  # Max sensor measurement range
    invalid_distance_start = 2.0        # Distance from which on pixels are started to set invalid with probability
    invalid_probability_low = 0.02      # Probability of invalid pixel at invalid_distance_start
    invalid_probability_high = 0.6      # Probability of invalid pixel at max_distance
    mean_bias = 0.02                    # Mean value of sensor bias over all distances$
    corner_error_factor = 1.43          # Factor 1.43 equals 42.5% higher error (mean 25%,60% from paper) for corners
    stdev_low = 0.005 / 1.35            # IQD to stdev approximation assuming normal distribution
    stdev_high = 0.08 / 1.35            # IQD to stdev approximation assuming normal distribution
    corner_1 = 0                        # Index of corner in ToF matrix
    corner_2 = 7                        # Index of corner in ToF matrix
    corner_3 = 56                       # Index of corner in ToF matrix
    corner_4 = 63                       # Index of corner in ToF matrix

    # Set all invalid pixels (inf in simulation) to 3m
    tof_image_processed[np.isinf(tof_image_processed)] = max_distance

    with np.nditer(tof_image_processed, flags=['f_index'], op_flags=['readwrite']) as it:
        for element in it:
            # Make pixels between invalid_distance_start and max_distance invalid (set to max_distance) in binary fashion
            # with increasing probability from 2%-->60% using linear interpolation between invalid_distance_start and max_distance
            if (element[...]  > invalid_distance_start and element[...]  < max_distance):
                invalid_probability = invalid_probability_low + (element[...] - invalid_distance_start) * \
                                      (invalid_probability_high - invalid_probability_low) / \
                                      (max_distance - invalid_distance_start)
                element[...] = np.random.choice([element[...], max_distance], p=[1-invalid_probability, invalid_probability])

            # Model sensor bias. Add gaussian noise with mean_bias and stdev linearly interpolate from stdev_low to stdev_high based on
            # distance between min_distance and max_distance. Corners have larger mean errors
            stdev_interpol = stdev_low + (element[...] - min_distance) * (stdev_high - stdev_low) / (max_distance - min_distance)
            if (it.index == corner_1 or it.index == corner_2 or it.index ==  corner_3 or it.index == corner_4):
                element[...] = element[...] + np.random.normal(mean_bias * corner_error_factor, stdev_interpol)
            else:
                element[...] = element[...] + np.random.normal(mean_bias, stdev_interpol)

    # Set all elements above max_distance to max_distance
    tof_image_processed = np.where(tof_image_processed > max_distance, max_distance, tof_image_processed)

    return tof_image_processed