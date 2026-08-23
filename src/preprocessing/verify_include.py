import os
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

MANIFEST = "data/include/metadata/include50_manifest.csv"

VIDEO_ROOT = r"D:\ISLR_DATA\INCLUDE\videos"


# --------------------------------------------------
# Load manifest
# --------------------------------------------------

print("Loading INCLUDE-50 manifest...")

df = pd.read_csv(MANIFEST)

print(f"Manifest entries: {len(df)}")


# --------------------------------------------------
# Verify every video
# --------------------------------------------------

found = []
missing = []


for _, row in df.iterrows():

    relative_path = row["video_path"]

    full_path = os.path.join(
        VIDEO_ROOT,
        relative_path
    )

    if os.path.isfile(full_path):

        found.append(relative_path)

    else:

        missing.append(relative_path)


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 50)
print("INCLUDE-50 VIDEO VERIFICATION")
print("=" * 50)

print(f"Required: {len(df)}")
print(f"Found:    {len(found)}")
print(f"Missing:  {len(missing)}")


# --------------------------------------------------
# Split statistics
# --------------------------------------------------

print()
print("Dataset split:")

if "split" in df.columns:

    print(
        df["split"]
        .value_counts()
        .to_string()
    )


# --------------------------------------------------
# Missing files
# --------------------------------------------------

if missing:

    print()
    print("Missing videos:")
    
    for path in missing:
        print(path)

else:

    print()
    print("ALL INCLUDE-50 VIDEOS FOUND.")


# --------------------------------------------------
# Final status
# --------------------------------------------------

print()
print("=" * 50)

if not missing:

    print("STATUS: READY FOR MEDIAPIPE")

else:

    print("STATUS: MISSING FILES")

print("=" * 50)