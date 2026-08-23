import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv

from normalize_landmarks import normalize_sequence


# ============================================================
# MediaPipe setup
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = "models/holistic_landmarker.task"


# ============================================================
# Dataset paths
# ============================================================

DATASET_DIR = "data/landmarks/custom"

CSV_PATH = os.path.join(
    DATASET_DIR,
    "labels.csv"
)


# ============================================================
# Landmark extraction
# ============================================================

def extract_landmarks(landmarks, count):

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

    left_hand = extract_landmarks(
        result.left_hand_landmarks,
        21
    )

    right_hand = extract_landmarks(
        result.right_hand_landmarks,
        21
    )

    pose = extract_landmarks(
        result.pose_landmarks,
        33
    )

    features = np.concatenate([
        left_hand,
        right_hand,
        pose
    ])

    return features.astype(np.float32)


# ============================================================
# Draw landmarks
# ============================================================

def draw_landmarks(frame, result):

    height, width = frame.shape[:2]

    # ----------------------------
    # Pose
    # ----------------------------

    if result.pose_landmarks:

        for landmark in result.pose_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if 0 <= x < width and 0 <= y < height:

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

    # ----------------------------
    # Left hand
    # ----------------------------

    if result.left_hand_landmarks:

        for landmark in result.left_hand_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if 0 <= x < width and 0 <= y < height:

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (255, 0, 0),
                    -1
                )

    # ----------------------------
    # Right hand
    # ----------------------------

    if result.right_hand_landmarks:

        for landmark in result.right_hand_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if 0 <= x < width and 0 <= y < height:

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1
                )


# ============================================================
# CSV setup
# ============================================================

def initialize_csv():

    os.makedirs(
        DATASET_DIR,
        exist_ok=True
    )

    if not os.path.exists(CSV_PATH):

        with open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "sample_id",
                "sign",
                "path",
                "frames"
            ])


# ============================================================
# Get next sample number
# ============================================================

def get_next_sample_number(sign):

    sign_dir = os.path.join(
        DATASET_DIR,
        sign
    )

    os.makedirs(
        sign_dir,
        exist_ok=True
    )

    existing = []

    for filename in os.listdir(sign_dir):

        if filename.startswith("sample_") and filename.endswith(".npy"):

            try:

                number = int(
                    filename[
                        7:-4
                    ]
                )

                existing.append(number)

            except ValueError:

                pass

    if not existing:

        return 1

    return max(existing) + 1


# ============================================================
# Record one sample
# ============================================================

def record_sample(
    landmarker,
    cap,
    sign,
    sample_number
):

    print()
    print("----------------------------------------")
    print(f"Sign: {sign}")
    print(f"Sample: {sample_number}")
    print("----------------------------------------")
    print("Press SPACE to START recording.")
    print("Perform the sign.")
    print("Press SPACE again to STOP.")
    print("Press Q to cancel.")
    print()

    sequence = []

    recording = False

    start_time = None




    while True:

        success, frame = cap.read()

        if not success:

            print("Could not read webcam frame.")

            return False


        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------
        # MediaPipe input
        # ----------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------
        # Timestamp
        # ----------------------------------------

        timestamp_ms = int(time.monotonic() * 1000)


        # ----------------------------------------
        # Detect
        # ----------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ----------------------------------------
        # If recording
        # ----------------------------------------

        if recording:

            features = extract_frame_features(
                result
            )

            sequence.append(features)


            elapsed = (
                time.time() - start_time
            )

            cv2.putText(
                frame,
                f"RECORDING  {elapsed:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "READY - Press SPACE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )


        # ----------------------------------------
        # Draw landmarks
        # ----------------------------------------

        draw_landmarks(
            frame,
            result
        )


        # ----------------------------------------
        # Display
        # ----------------------------------------

        cv2.imshow(
            "ISL Dataset Recorder",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------
        # SPACE
        # ----------------------------------------

        if key == 32:

            if not recording:

                # Start

                recording = True

                sequence = []

                start_time = time.time()

                print("Recording started...")

            else:

                # Stop

                recording = False

                print("Recording stopped.")

                break


        # ----------------------------------------
        # Q
        # ----------------------------------------

        if key == ord("q"):

            print("Recording cancelled.")

            return False


    # ========================================================
    # Validate
    # ========================================================

    if len(sequence) < 5:

        print(
            "Recording too short. Not saved."
        )

        return False


    # ========================================================
    # Convert to NumPy
    # ========================================================

    sequence = np.array(
        sequence,
        dtype=np.float32
    )


    print(
        f"Raw sequence shape: {sequence.shape}"
    )


    # ========================================================
    # Normalize
    # ========================================================

    normalized = normalize_sequence(
        sequence
    )


    print(
        f"Normalized shape: {normalized.shape}"
    )


    # ========================================================
    # Save
    # ========================================================

    sign_dir = os.path.join(
        DATASET_DIR,
        sign
    )

    os.makedirs(
        sign_dir,
        exist_ok=True
    )


    filename = (
        f"sample_{sample_number:03d}.npy"
    )


    file_path = os.path.join(
        sign_dir,
        filename
    )


    np.save(
        file_path,
        normalized
    )


    # ========================================================
    # Relative path for CSV
    # ========================================================

    relative_path = os.path.relpath(
        file_path
    )

    # Use forward slashes
    relative_path = relative_path.replace(
        "\\",
        "/"
    )


    # ========================================================
    # Update CSV
    # ========================================================

    with open(
        CSV_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            f"{sign}_{sample_number:03d}",
            sign,
            relative_path,
            len(normalized)
        ])


    print()
    print("========================================")
    print("Sample saved successfully!")
    print("========================================")
    print(f"Sign:   {sign}")
    print(f"File:   {file_path}")
    print(f"Frames: {len(normalized)}")
    print(f"Shape:  {normalized.shape}")
    print()


    return True


# ============================================================
# Main
# ============================================================

def main():

    initialize_csv()


    # ----------------------------------------
    # Ask for sign
    # ----------------------------------------

    sign = input(
        "Enter sign name (example: HELLO): "
    ).strip().upper()


    if not sign:

        print("Sign name cannot be empty.")

        return


    # Replace spaces
    sign = sign.replace(
        " ",
        "_"
    )


    # ----------------------------------------
    # Number of recordings
    # ----------------------------------------

    try:

        count = int(
            input(
                "How many samples? "
            )
        )

    except ValueError:

        print(
            "Please enter a valid number."
        )

        return


    if count <= 0:

        print(
            "Number of samples must be greater than 0."
        )

        return


    # ----------------------------------------
    # Sample numbering
    # ----------------------------------------

    next_number = get_next_sample_number(
        sign
    )


    # ----------------------------------------
    # MediaPipe options
    # ----------------------------------------

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


    # ----------------------------------------
    # Webcam
    # ----------------------------------------

    cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return


    # ----------------------------------------
    # Start MediaPipe
    # ----------------------------------------

    with HolisticLandmarker.create_from_options(
        options
    ) as landmarker:


        for i in range(count):

            sample_number = (
                next_number + i
            )


            success = record_sample(
                landmarker,
                cap,
                sign,
                sample_number
            )


            if not success:

                print(
                    "Skipping this sample."
                )


            # Small pause between recordings

            print(
                "Get ready for the next sample..."
            )

            time.sleep(1)


    cap.release()

    cv2.destroyAllWindows()


    print()
    print("========================================")
    print("DATASET RECORDING COMPLETE")
    print("========================================")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Labels:  {CSV_PATH}")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()