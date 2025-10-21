import cv2
import pyttsx3
from deuteranopia_sim import simulate_deuteranopia
from gesture_control import detect_gesture
from color_adjustment import detect_color

engine = pyttsx3.init()

def speak_color(color):
    engine.say(f"The color is {color}")
    engine.runAndWait()

def main():
    cap = cv2.VideoCapture(0)
    last_color = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deuteranopia Simulation
        simulated = simulate_deuteranopia(frame)

        # Detect Colors
        detected_colors = detect_color(frame)

        # Detect Gesture
        gesture = detect_gesture(frame)

        # Speak detected color
        if gesture and detected_colors:
            color_to_speak = detected_colors[0]
            if color_to_speak != last_color:
                speak_color(color_to_speak)
                last_color = color_to_speak

        # Display overlay
        text = f"Detected: {', '.join(detected_colors)}" if detected_colors else "No color detected"
        cv2.putText(simulated, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        cv2.imshow("Normal View", frame)
        cv2.imshow("Deuteranopia Simulation", simulated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
