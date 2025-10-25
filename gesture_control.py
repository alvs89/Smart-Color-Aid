import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

gesture_counter = 0
gesture_frames = 15  # frames to hold gesture before triggering

def index_raised_condition(handLms):
    """True if index finger is up and all other fingers are down."""
    return (handLms.landmark[8].y < handLms.landmark[6].y and  # index up
            handLms.landmark[12].y > handLms.landmark[10].y and  # middle down
            handLms.landmark[16].y > handLms.landmark[14].y and  # ring down
            handLms.landmark[20].y > handLms.landmark[18].y)      # pinky down

def open_raised_condition(handLms):
    """True if index finger is up and all other fingers are down."""
    return (handLms.landmark[8].y < handLms.landmark[6].y and  # index up
            handLms.landmark[12].y < handLms.landmark[10].y and  # middle up
            handLms.landmark[16].y < handLms.landmark[14].y and  # ring up
            handLms.landmark[20].y < handLms.landmark[18].y)      # pinky up
    
def thumb_up_condition(handLms, threshold=0.02):
    """Detects thumbs-up more reliably even if fingers are slightly bent."""
    thumb_tip = handLms.landmark[4]
    thumb_ip = handLms.landmark[3]

    index_tip = handLms.landmark[8]
    index_pip = handLms.landmark[6]
    middle_tip = handLms.landmark[12]
    middle_pip = handLms.landmark[10]
    ring_tip = handLms.landmark[16]
    ring_pip = handLms.landmark[14]
    pinky_tip = handLms.landmark[20]
    pinky_pip = handLms.landmark[18]

    # Thumb up
    thumb_up = thumb_tip.y + threshold < thumb_ip.y

    # Other fingers not fully extended
    index_folded = index_tip.y > index_pip.y - threshold
    middle_folded = middle_tip.y > middle_pip.y - threshold
    ring_folded = ring_tip.y > ring_pip.y - threshold
    pinky_folded = pinky_tip.y > pinky_pip.y - threshold

    return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded

def detect_gesture(frame):
    """
    Returns 'index', 'thumb', or None.
    Uses gesture_counter to trigger only after holding the gesture.
    """
    global gesture_counter
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        if index_raised_condition(handLms):
            gesture_counter += 1
            if gesture_counter >= gesture_frames:
                gesture_counter = 0
                return 'index'
        elif thumb_up_condition(handLms):
            gesture_counter += 1
            if gesture_counter >= gesture_frames:
                gesture_counter = 0
                return 'thumb'
        elif open_raised_condition(handLms):
            gesture_counter += 1
            if gesture_counter >= gesture_frames:
                gesture_counter = 0
                return 'open'
        else:
            gesture_counter = 0
    else:
        gesture_counter = 0

    return None
