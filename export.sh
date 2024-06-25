#!/bin/bash

APP_BUNDLE_ID="com.aisport.coach.SportsCoach" # Replace with your app's bundle identifier
EXPORT_PATH="$HOME/Downloads/tmp" # Set the correct export path
TARGET_PATH="$HOME/golf/video-diff/tmp/ios" # Set the target path for .json files

# Get the device's UDID
DEVICE_ID=$(idevice_id -l | head -n 1)

if [ -z "$DEVICE_ID" ]; then
  echo "No device found"
  exit 1
fi

# Create export path if it doesn't exist
mkdir -p "$EXPORT_PATH"

# Create target path if it doesn't exist
mkdir -p "$TARGET_PATH"

# Check if the app is installed on the device
ios-deploy --id "$DEVICE_ID" --exists --bundle_id "$APP_BUNDLE_ID"

if [ $? -ne 0 ]; then
  echo "App with bundle ID $APP_BUNDLE_ID not found on device $DEVICE_ID"
  exit 1
fi

# Export the tmp folder using ios-deploy
ios-deploy --id "$DEVICE_ID" --download=/tmp --to="$EXPORT_PATH" --bundle_id "$APP_BUNDLE_ID"

if [ $? -ne 0 ]; then
  echo "Failed to export tmp folder to: $EXPORT_PATH"
  exit 1
fi

echo "tmp folder exported successfully to: $EXPORT_PATH"

# Move all .json files from EXPORT_PATH/tmp to TARGET_PATH
find "$EXPORT_PATH/tmp" -name "*.json" -exec mv {} "$TARGET_PATH" \;

if [ $? -ne 0 ]; then
  echo "Failed to move .json files to: $TARGET_PATH"
  exit 1
fi

echo ".json files moved successfully to: $TARGET_PATH"

# Remove EXPORT_PATH if mv command was successful
rm -rf "$EXPORT_PATH"

if [ $? -ne 0 ]; then
  echo "Failed to remove export path: $EXPORT_PATH"
  exit 1
fi

echo "Export path $EXPORT_PATH removed successfully"
