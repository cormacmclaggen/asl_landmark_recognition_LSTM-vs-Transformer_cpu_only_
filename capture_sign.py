import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
# ----------------------------------------------------------
os.makedirs("data_parquet", exist_ok=True)
# Mediapipe setup
# ----------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# ----------------------------------------------------------
# Create a dataframe for each frame's landmarks
# ----------------------------------------------------------
def create_frame_landmark_df(results, frame, xyz):
    """
    Takes the results from mediapipe and creates a dataframe of the landmarks
    inputs:
        results: mediapipe results object
        frame: frame number
        xyz: dataframe of the xyz example data
    """
    xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()

    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    # FACE
    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    # POSE
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    # LEFT HAND
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    # RIGHT HAND
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    # Add labels & index
    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    # Combine all
    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks


# ----------------------------------------------------------
# Main capture loop
# ----------------------------------------------------------
def do_capture_loop(xyz):
    all_landmarks = []
    # try:
    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        frame = 0
        while cap.isOpened():
            frame += 1

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Create dataframe for this frame
            landmarks = create_frame_landmark_df(results, frame, xyz)
            all_landmarks.append(landmarks)

            # Draw landmarks on image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())

            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # except:
    #     return all_landmarks

    return all_landmarks


# ----------------------------------------------------------
# Save when run directly
# ----------------------------------------------------------
if __name__ == "__main__":
    pq_file = '/home/lananh/GISLR/train_landmark_files/16069/695046.parquet'
    xyz = pd.read_parquet(pq_file)
    landmarks = do_capture_loop(xyz)

    if not landmarks:
        print("No landmarks captured. Check your camera.")
    else:
        output_file = f"data_parquet/capture_{int(time.time())}.parquet"
        pd.concat(landmarks).reset_index(drop=True).to_parquet(output_file)
    
    print("Saved to:", output_file)