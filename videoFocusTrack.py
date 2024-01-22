import cv2
import dlib
import time

# Initialize the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Python Programs/speech and video proj/resources/shape_predictor_68_face_landmarks.dat")

# Open the webcam
cap = cv2.VideoCapture(0)

# Perform initial calibration
def perform_calibration():
    scale = 1.5  # Replace with your calibration logic
    return scale

# Perform initial calibration
scale = perform_calibration()

# Initialize blink counter
blink_counter = 0

# Initialize focus timer variables
focus_start_time = None
focus_duration = 0
focus_threshold = 5  # Adjust this value as needed

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Check if there are any faces detected
    if len(faces) > 0 :
        # Start or continue focus timer
        if focus_start_time is None:
            focus_start_time = time.time()
        else:
            focus_duration = time.time() - focus_start_time
    else:
        # Reset focus timer and play sound
        if focus_duration >= focus_threshold:
            print("Play sound")
        focus_start_time = None
        focus_duration = 0

    for face in faces:
        # Detect landmarks for the face
        landmarks = predictor(gray, face)

        # Calculate the bounding box for the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Detect eyes
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)

        # Calculate the center of each eyelid
        top_left_eyelid_center = (landmarks.part(37).x, landmarks.part(37).y)
        bottom_left_eyelid_center = (landmarks.part(41).x, landmarks.part(41).y)
        top_right_eyelid_center = (landmarks.part(44).x, landmarks.part(44).y)
        bottom_right_eyelid_center = (landmarks.part(46).x, landmarks.part(46).y)

        # Calculate the center of each eye
        left_eye_center = (left_eye.x, left_eye.y)
        right_eye_center = (right_eye.x, right_eye.y)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw circles around the eyes
        cv2.circle(frame, left_eye_center, 2, (0, 0, 255), -1)
        cv2.circle(frame, right_eye_center, 2, (0, 0, 255), -1)
        
        # Draw circles around the eyelids
        cv2.circle(frame, top_left_eyelid_center, 2, (255, 255, 255), -1)
        cv2.circle(frame, bottom_left_eyelid_center, 2, (255, 255, 255), -1)
        cv2.circle(frame, top_right_eyelid_center, 2, (255, 255, 255), -1)
        cv2.circle(frame, bottom_right_eyelid_center, 2, (255, 255, 255), -1)

        # Detect pupils
        left_pupil = landmarks.part(37)
        right_pupil = landmarks.part(46)

        # Calculate the center of each pupil
        left_pupil_center = (left_pupil.x, left_pupil.y)
        right_pupil_center = (right_pupil.x, right_pupil.y)

    # Display the focus time in the video window
    cv2.putText(frame, f"Focus Time: {focus_duration:.2f} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Window Name", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()