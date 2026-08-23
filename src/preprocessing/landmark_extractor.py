import cv2
import mediapipe as mp
import numpy as np
import time


# --------------------------------------------------
# MediaPipe
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions

HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = "models/holistic_landmarker.task"


# --------------------------------------------------
# Landmark extraction helpers
# --------------------------------------------------

def extract_landmarks(landmarks, count):
    """
    Convert MediaPipe landmarks into a flat
    [x, y, z] feature vector.

    If landmarks are missing, return zeros.
    """

    if not landmarks:

        return np.zeros(
            count * 3,
            dtype=np.float32
        )

    values = []

    for landmark in landmarks:

        values.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(
        values,
        dtype=np.float32
    )


def extract_frame_features(result):
    """
    Convert one MediaPipe result into
    one fixed-size feature vector.
    """

    # ----------------------------------------------
    # Left hand
    # ----------------------------------------------

    left_hand = extract_landmarks(
        result.left_hand_landmarks,
        21
    )


    # ----------------------------------------------
    # Right hand
    # ----------------------------------------------

    right_hand = extract_landmarks(
        result.right_hand_landmarks,
        21
    )


    # ----------------------------------------------
    # Pose
    # ----------------------------------------------

    pose = extract_landmarks(
        result.pose_landmarks,
        33
    )


    # ----------------------------------------------
    # Combine
    # ----------------------------------------------

    features = np.concatenate([
        left_hand,
        right_hand,
        pose
    ])


    return features


# --------------------------------------------------
# Webcam recorder
# --------------------------------------------------

def record_landmarks(
    output_path,
    duration=5
):

    options = HolisticLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),

        running_mode=RunningMode.VIDEO,

        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,

        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,

        min_hand_landmarks_confidence=0.5
    )


    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )


    sequence = []


    with HolisticLandmarker.create_from_options(
        options
    ) as landmarker:

        print()
        print("Get ready...")
        time.sleep(2)

        print("Recording started!")
        print("Perform your sign.")

        start_time = time.time()

        while True:

            success, frame = cap.read()

            if not success:

                break


            # --------------------------------------
            # RGB
            # --------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )


            # --------------------------------------
            # Timestamp
            # --------------------------------------

            timestamp_ms = int(
                (time.time() - start_time)
                * 1000
            )


            # --------------------------------------
            # MediaPipe
            # --------------------------------------

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )


            # --------------------------------------
            # Extract features
            # --------------------------------------

            features = extract_frame_features(
                result
            )

            sequence.append(features)


            # --------------------------------------
            # Display
            # --------------------------------------

            elapsed = time.time() - start_time

            cv2.putText(
                frame,
                f"Recording: {elapsed:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "ISL Landmark Recorder",
                frame
            )


            # --------------------------------------
            # Stop
            # --------------------------------------

            if elapsed >= duration:

                break


            if cv2.waitKey(1) & 0xFF == ord("q"):

                break


    cap.release()
    cv2.destroyAllWindows()


    # ------------------------------------------
    # Convert to NumPy
    # ------------------------------------------

    sequence = np.array(
        sequence,
        dtype=np.float32
    )


    # ------------------------------------------
    # Save
    # ------------------------------------------

    np.save(
        output_path,
        sequence
    )


    print()
    print("Recording finished.")
    print("Saved:", output_path)
    print("Shape:", sequence.shape)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    record_landmarks(
        "data/landmarks/test_sample.npy",
        duration=5
    )