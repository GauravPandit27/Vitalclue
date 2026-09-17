"""
VitalCue - Vision Module
Handles MediaPipe Pose (respiratory signal) and Face Mesh (expression signal).
"""
import cv2
import mediapipe as mp

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


class VisionProcessor:
    """Extracts raw respiratory and facial signals from a video frame."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr):
        """
        Returns:
            annotated_frame: frame with pose/face landmarks drawn (for live UI feedback,
                              so the person can visually confirm lock-on)
            shoulder_y: average normalized Y of left+right shoulder (float or None)
            face_landmarks: raw face mesh landmarks (or None) for expression scoring
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        pose_results = self.pose.process(rgb)
        face_results = self.face_mesh.process(rgb)

        rgb.flags.writeable = True
        annotated = frame_bgr.copy()

        import math
        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        shoulder_y = None
        face_landmarks = None
        alignment_feedback = "ALIGNED"
        expression = "Normal"
        gesture = "Normal"

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            left = lm[LEFT_SHOULDER]
            right = lm[RIGHT_SHOULDER]
            nose = lm[0]
            l_wrist = lm[15]
            r_wrist = lm[16]
            
            # Gesture Detection (Hand to Face)
            if (l_wrist.visibility > 0.5 and dist(nose, l_wrist) < 0.2) or (r_wrist.visibility > 0.5 and dist(nose, r_wrist) < 0.2):
                gesture = "Hand on Face"
            
            if left.visibility > 0.5 and right.visibility > 0.5:
                shoulder_y = (left.y + right.y) / 2.0
                shoulder_dist = abs(left.x - right.x)
                
                if shoulder_y > 0.85:
                    alignment_feedback = "Move back so your chest is visible."
                elif shoulder_dist > 0.7:
                    alignment_feedback = "Too close. Move back slightly."
                elif shoulder_dist < 0.15:
                    alignment_feedback = "Too far. Move closer to the camera."
            else:
                alignment_feedback = "Ensure both shoulders are clearly visible."

            custom_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            self.mp_drawing.draw_landmarks(
                annotated, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=custom_spec, connection_drawing_spec=custom_spec
            )
        else:
            alignment_feedback = "No person detected in frame."

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # Expression Detection
            left_eye_inner = face_landmarks.landmark[133]
            right_eye_inner = face_landmarks.landmark[362]
            mouth_left = face_landmarks.landmark[61]
            mouth_right = face_landmarks.landmark[291]
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            eye_distance = dist(left_eye_inner, right_eye_inner)
            mouth_width = dist(mouth_left, mouth_right)
            mouth_height = dist(upper_lip, lower_lip)

            if mouth_width > 0 and eye_distance > 0:
                if mouth_height / mouth_width > 0.5:
                    expression = "Surprised"
                elif mouth_width / eye_distance > 1.8:
                    expression = "Smiling"
                elif mouth_height / mouth_width < 0.15 and mouth_width / eye_distance < 1.5:
                    expression = "Neutral"
                else:
                    expression = "Angry"
                
            custom_face_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            self.mp_drawing.draw_landmarks(
                annotated, face_landmarks, self.mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=custom_face_spec, connection_drawing_spec=custom_face_spec
            )

        return annotated, shoulder_y, face_landmarks, alignment_feedback, expression, gesture

    def close(self):
        self.pose.close()
        self.face_mesh.close()
