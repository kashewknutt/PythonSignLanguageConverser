import cv2
from detect_hand import detect_hand

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect hand in the frame using detect_hand() function
        frame_with_hand = detect_hand(frame)

        # Display the frame with detected hand
        cv2.imshow('Hand Detection', frame_with_hand)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
