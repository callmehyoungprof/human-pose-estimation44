"""Real-time human pose estimation using OpenCV and MediaPipe.

Run with: python pose_estimation.py
Press q or Esc to quit.
"""

import math
import time

import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """Return the angle ABC in degrees using normalized landmark coordinates."""
    if a is None or b is None or c is None:
        return None

    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x)
        - math.atan2(a.y - b.y, a.x - b.x)
    )
    return round(abs(angle) if abs(angle) <= 180 else 360 - abs(angle))


def visible_angle(landmarks, first, middle, last):
    """Calculate an angle only when all three landmarks are visible."""
    points = [landmarks[point] for point in (first, middle, last)]
    if any(point.visibility < 0.5 for point in points):
        return None
    return calculate_angle(*points)


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions or camera index."
        )

    previous_time = time.time()

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                # Mirror the frame for a more natural webcam view.
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=(245, 245, 245), thickness=2, circle_radius=2
                        ),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(80, 220, 160), thickness=3, circle_radius=2
                        ),
                    )

                    points = results.pose_landmarks.landmark
                    names = [
                        ("Left elbow", mp_pose.PoseLandmark.LEFT_SHOULDER,
                         mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                        ("Right elbow", mp_pose.PoseLandmark.RIGHT_SHOULDER,
                         mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                        ("Left knee", mp_pose.PoseLandmark.LEFT_HIP,
                         mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                        ("Right knee", mp_pose.PoseLandmark.RIGHT_HIP,
                         mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    ]

                    y = 35
                    for label, first, middle, last in names:
                        angle = visible_angle(points, first, middle, last)
                        value = f"{angle} deg" if angle is not None else "N/A"
                        cv2.putText(
                            frame, f"{label}: {value}", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA,
                        )
                        y += 28
                else:
                    cv2.putText(
                        frame, "No pose detected - move into camera view",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 255), 2, cv2.LINE_AA,
                    )

                now = time.time()
                fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (15, frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Human Pose Estimation - press Q to quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
