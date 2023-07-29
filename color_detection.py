import cv2
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def read_color_from_image(img_path):
    img = cv2.imread(img_path)
    return img

def read_color_from_webcam():
    cap = cv2.VideoCapture(0)

    index = ["color_name", "hex", "R", "G", "B"]
    csv = pd.read_csv('colors.csv', names=index, header=None)

    # Extract features (RGB values) and labels (color names) from the CSV data
    features = csv[['R', 'G', 'B']].values
    labels = csv['color_name'].values

    # Create and train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features, labels)

    def process_mouse_click(event, x, y, flags, param):
        nonlocal clicked, r, g, b, last_color_info
        if event == cv2.EVENT_LBUTTONDBLCLK:
            clicked = True
            b, g, r = frame[y, x]
            b, g, r = int(b), int(g), int(r)

            # Predict the color name using KNN based on the RGB values
            test_data = np.array([[r, g, b]])
            color_name = knn.predict(test_data)[0]

            last_color_info = f"{color_name} R={r} G={g} B={b}"

    cv2.namedWindow('Webcam')
    cv2.setMouseCallback('Webcam', process_mouse_click)

    clicked = False
    r = g = b = 0
    last_color_info = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if clicked:
            clicked_frame = frame.copy()

            cv2.rectangle(clicked_frame, (20, 20), (750, 60), (b, g, r), -1)
            cv2.putText(clicked_frame, last_color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # For very light colors, display text in black for better visibility
            if r + g + b >= 600:
                cv2.putText(clicked_frame, last_color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow("Webcam", clicked_frame)
            clicked = False
        else:
            # Display the last clicked color information in each frame until a new color is clicked
            if last_color_info:
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, (20, 20), (750, 60), (b, g, r), -1)
                cv2.putText(frame_copy, last_color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                if r + g + b >= 600:
                    cv2.putText(frame_copy, last_color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("Webcam", frame_copy)
            else:
                cv2.imshow("Webcam", frame)

        # Check for 'Esc' key press to close the webcam window
        key = cv2.waitKey(1)
        if key & 0xFF == 27:  # 27 is the ASCII code for 'Esc' key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    option = input("Select an option (a or b):\n"
                   "a. Read color from image\n"
                   "b. Read color from webcam\n"
                   "Your choice: ")

    if option.lower() == 'a':
        img_path = "colorpic.jpg"
        img = read_color_from_image(img_path)

        index = ["color_name", "hex", "R", "G", "B"]
        csv = pd.read_csv('colors.csv', names=index, header=None)

        # Extract features (RGB values) and labels (color names) from the CSV data
        features = csv[['R', 'G', 'B']].values
        labels = csv['color_name'].values

        # Create and train a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(features, labels)

        cv2.namedWindow('color detection')
        cv2.setMouseCallback('color detection', lambda event, x, y, flags, param: process_mouse_click(event, x, y))

        while True:
            cv2.imshow("color detection", img)

            # Exit the loop if the 'Esc' key is pressed
            if cv2.waitKey(20) & 0xFF == 27:
                break

    elif option.lower() == 'b':
        read_color_from_webcam()

    else:
        print("Invalid option. Please choose either 'a' or 'b'.")
