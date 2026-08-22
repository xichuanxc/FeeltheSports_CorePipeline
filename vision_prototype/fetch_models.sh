#!/bin/bash
# Fetch the MediaPipe PoseLandmarker task files the prototype needs.
# Kept out of git: they are 5 to 29 MB of published weights, not source.
set -e
cd "$(dirname "$0")"
mkdir -p models
BASE=https://storage.googleapis.com/mediapipe-models/pose_landmarker
for m in lite full heavy; do
  f="models/pose_landmarker_${m}.task"
  if [ -f "$f" ]; then echo "have $f"; continue; fi
  echo "fetching $f"
  curl -fsSL -o "$f" "$BASE/pose_landmarker_${m}/float16/latest/pose_landmarker_${m}.task"
done
