from deepface import DeepFace
import cv2
import os

# Directory path containing known faces
known_faces_dir = "C:\\Users\\SUBROTO MONDAL\\PycharmProjects\\pythonProject2\\known_faces"  # Replace with your directory path

# Check if the directory exists
if not os.path.isdir(known_faces_dir):
    print(f"Directory at {known_faces_dir} not found.")
    exit()

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Save the captured frame as a temporary image
    temp_frame_path = "temp_frame.jpg"
    cv2.imwrite(temp_frame_path, frame)

    # Initialize result variables
    text = "No Match"
    color = (0, 0, 255)  # Red color for no match
    match_found = False

    # Loop through each image in known_faces_dir
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            face_path = os.path.join(known_faces_dir, filename)

            # Perform face recognition on the current frame
            try:
                result = DeepFace.verify(
                    img1_path=face_path,
                    img2_path=temp_frame_path,
                    model_name='VGG-Face', # you can change models here(Facenet,ArcFace,OpenFace,DeepFace,DeepID)
                    enforce_detection=True
                )

                # If a match is found, set match result and color
                if result["verified"]:
                    text = f"Match Found: {os.path.splitext(filename)[0]}"
                    color = (0, 255, 0)  # Green color for match
                    match_found = True
                    break  # Stop checking once a match is found

            except Exception:
                # Do nothing and continue with the next image in known_faces_dir
                continue

    # Display text on the frame
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the current frame with the verification result
    cv2.imshow("Continuous Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Delete the temporary frame image after processing
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
