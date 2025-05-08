import face_recognition
import cv2
import os

# Load known faces from a directory
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            known_image = face_recognition.load_image_file(image_path)
            known_face_encoding = face_recognition.face_encodings(known_image)[0]
            known_face_encodings.append(known_face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names

# Recognize faces from the webcam feed
def recognize_faces():
    known_faces_dir = "C:\\Users\\SUBROTO MONDAL\\PycharmProjects\\pythonProject2\\known_faces"
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if face_locations:  # Only process if faces are detected
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                color = (0, 0, 255)  # Default color for unrecognized faces (Red)

                # Use the first matched known face
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    color = (0, 255, 0)  # Color for recognized faces (Green)

                # Draw a rectangle around the face and add a label
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the resulting frame
        cv2.imshow("Face Recognition", frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run face recognition
recognize_faces()

