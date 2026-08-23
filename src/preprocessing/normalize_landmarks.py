import numpy as np


# --------------------------------------------------
# Feature layout
# --------------------------------------------------

LEFT_HAND_START = 0
LEFT_HAND_END = 63

RIGHT_HAND_START = 63
RIGHT_HAND_END = 126

POSE_START = 126
POSE_END = 225


# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_landmark(features, start, index):
    """
    Get one XYZ landmark from a feature vector.
    """

    i = start + index * 3

    return features[i:i + 3]


def set_landmark(features, start, index, value):
    """
    Write one XYZ landmark into a feature vector.
    """

    i = start + index * 3

    features[i:i + 3] = value


def is_valid_landmark(point):
    """
    A missing landmark is represented by [0, 0, 0].
    """

    return not np.allclose(point, 0.0)


# --------------------------------------------------
# Normalize one frame
# --------------------------------------------------

def normalize_frame(frame):
    """
    Normalize one frame using the midpoint
    between the shoulders as the origin.

    Missing landmarks remain [0, 0, 0].
    """

    frame = frame.copy()


    # ----------------------------------------------
    # Get shoulders
    # ----------------------------------------------

    left_shoulder = get_landmark(
        frame,
        POSE_START,
        LEFT_SHOULDER
    )

    right_shoulder = get_landmark(
        frame,
        POSE_START,
        RIGHT_SHOULDER
    )


    # ----------------------------------------------
    # Cannot normalize without shoulders
    # ----------------------------------------------

    if (
        not is_valid_landmark(left_shoulder)
        or
        not is_valid_landmark(right_shoulder)
    ):
        return frame


    # ----------------------------------------------
    # Shoulder center
    # ----------------------------------------------

    center = (
        left_shoulder + right_shoulder
    ) / 2.0


    # ----------------------------------------------
    # Shoulder distance = scale
    # ----------------------------------------------

    scale = np.linalg.norm(
        left_shoulder - right_shoulder
    )


    # Avoid division by zero
    if scale < 1e-6:
        return frame


    # ----------------------------------------------
    # Normalize all landmark groups
    # ----------------------------------------------

    groups = [
        (LEFT_HAND_START, LEFT_HAND_END),
        (RIGHT_HAND_START, RIGHT_HAND_END),
        (POSE_START, POSE_END)
    ]


    for start, end in groups:

        number_of_landmarks = (
            end - start
        ) // 3


        for index in range(
            number_of_landmarks
        ):

            point = get_landmark(
                frame,
                start,
                index
            )


            # Don't modify missing landmarks
            if not is_valid_landmark(point):
                continue


            normalized = (
                point - center
            ) / scale


            set_landmark(
                frame,
                start,
                index,
                normalized
            )


    return frame


# --------------------------------------------------
# Normalize complete sequence
# --------------------------------------------------

def normalize_sequence(sequence):

    normalized = np.zeros_like(
        sequence,
        dtype=np.float32
    )


    for i, frame in enumerate(sequence):

        normalized[i] = normalize_frame(
            frame
        )


    return normalized


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    input_path = (
        "data/landmarks/"
        "test_sample.npy"
    )

    output_path = (
        "data/landmarks/"
        "test_sample_normalized.npy"
    )


    # Load
    sequence = np.load(
        input_path
    )


    print(
        "Original shape:",
        sequence.shape
    )


    # Normalize
    normalized = normalize_sequence(
        sequence
    )


    # Save
    np.save(
        output_path,
        normalized
    )


    print(
        "Normalized shape:",
        normalized.shape
    )


    print(
        "Normalized dtype:",
        normalized.dtype
    )


    print(
        "Normalized min:",
        normalized.min()
    )


    print(
        "Normalized max:",
        normalized.max()
    )


    print(
        "Normalized mean:",
        normalized.mean()
    )


    print(
        "Normalized std:",
        normalized.std()
    )


    print(
        "Saved:",
        output_path
    )