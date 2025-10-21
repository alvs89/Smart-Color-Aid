import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

gesture_counter = 0

def detect_gesture(frame):
    """
    Detects if index finger is raised.
    Requires the gesture to be held for a few frames to trigger.
    """
    global gesture_counter
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            index_tip = handLms.landmark[8]
            index_pip = handLms.landmark[6]

            # Gesture smoothing (must hold for at least 3 frames)
            if index_tip.y < index_pip.y:
                gesture_counter += 1
                if gesture_counter > 3:
                    gesture_counter = 0
                    return True
            else:
                gesture_counter = 0
    return False
