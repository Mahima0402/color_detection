import cv2
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class ColorDetector:
    def __init__(self):
        self.knn = None

    def read_color_from_image(self, img_path):
        img = cv2.imread(img_path)
        return img

    def read_color_from_webcam(self):
        self.initialize_knn()
        cap = cv2.VideoCapture(0)

        clicked = False
        r = g = b = 0
        last_color_info = ""

        def process_mouse_click(event, x, y, flags, param):
            nonlocal clicked, r, g, b, last_color_info
            if event == cv2.EVENT_LBUTTONDBLCLK:
                clicked = True
                b, g, r = frame[y, x]
                b, g, r = int(b), int(g), int(r)

                # Predict the color name using KNN based on the RGB values
                test_data = np.array([[r, g, b]])
                color_name = self.knn.predict(test_data)[0]

                last_color_info = f"{color_name} R={r} G={g} B={b}"

                # Display the color information on the webcam window
                cv2.rectangle(frame, (20, 20), (750, 60), (b, g, r), -1)
                cv2.putText(frame, last_color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                # For very light colors, display text in black for better visibility
                if r + g + b >= 600:
                    cv2.putText(frame, last_color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.namedWindow('Webcam')
        cv2.setMouseCallback('Webcam', process_mouse_click)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if clicked:
                cv2.imshow("Webcam", frame)
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

    def take_live_picture(self):
        self.initialize_knn()
        cap = cv2.VideoCapture(0)

        def process_mouse_click(event, x, y, flags, param):
            nonlocal clicked, r, g, b, last_color_info, frame
            if event == cv2.EVENT_LBUTTONDBLCLK:
                clicked = True
                b, g, r = frame[y, x]
                b, g, r = int(b), int(g), int(r)

                # Predict the color name using KNN based on the RGB values
                test_data = np.array([[r, g, b]])
                color_name = self.knn.predict(test_data)[0]

                last_color_info = f"{color_name} R={r} G={g} B={b}"

        clicked = False
        r = g = b = 0
        last_color_info = ""

        countdown = 3
        while countdown > 0:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, str(countdown), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 10)
            cv2.imshow("Webcam", frame)

            cv2.waitKey(1000)  # Wait for 1 second
            countdown -= 1

        if countdown == 0:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Live Picture", frame)
                cv2.setMouseCallback('Live Picture', process_mouse_click)  # Update the callback function

                while True:
                    key = cv2.waitKey(1)

                    # Press 'Esc' to close both windows
                    if key & 0xFF == 27:
                        break

                    # Display the color information on the taken picture window after the 'Esc' key is pressed
                    if last_color_info:
                        frame_copy = frame.copy()
                        cv2.rectangle(frame_copy, (20, 20), (750, 60), (b, g, r), -1)
                        cv2.putText(frame_copy, last_color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        if r + g + b >= 600:
                            cv2.putText(frame_copy, last_color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.imshow("Live Picture", frame_copy)

                cv2.destroyAllWindows()

        cap.release()

    def detect_color(self, img):
        self.initialize_knn()
        clicked = False
        r = g = b = 0
        last_color_info = ""

        def process_mouse_click(event, x, y, flags, param):
            nonlocal clicked, r, g, b, last_color_info
            if event == cv2.EVENT_LBUTTONDBLCLK:
                clicked = True
                b, g, r = img[y, x]
                b, g, r = int(b), int(g), int(r)

                # Predict the color name using KNN based on the RGB values
                test_data = np.array([[r, g, b]])
                color_name = self.knn.predict(test_data)[0]

                last_color_info = f"{color_name} R={r} G={g} B={b}"

        cv2.namedWindow('Color Detection')
        cv2.setMouseCallback('Color Detection', process_mouse_click)

        while True:
            if clicked:
                cv2.imshow("Color Detection", img)
                clicked = False
            else:
                # Display the last clicked color information in each frame until a new color is clicked
                if last_color_info:
                    frame_copy = img.copy()
                    cv2.rectangle(frame_copy, (20, 20), (750, 60), (b, g, r), -1)
                    cv2.putText(frame_copy, last_color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    if r + g + b >= 600:
                        cv2.putText(frame_copy, last_color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Color Detection", frame_copy)
                else:
                    cv2.imshow("Color Detection", img)

            # Check for 'Esc' key press to close the window
            key = cv2.waitKey(1)
            if key & 0xFF == 27:  # 27 is the ASCII code for 'Esc' key
                break

        cv2.destroyAllWindows()

    def initialize_knn(self):
        if self.knn is None:
            index = ["color_name", "hex", "R", "G", "B"]
            csv = pd.read_csv('colors.csv', names=index, header=None)

            # Extract features (RGB values) and labels (color names) from the CSV data
            features = csv[['R', 'G', 'B']].values
            labels = csv['color_name'].values

            # Create and train a KNN classifier
            self.knn = KNeighborsClassifier(n_neighbors=1)
            self.knn.fit(features, labels)

if __name__ == "__main__":
    color_detector = ColorDetector()

    option = input("Select an option (a, b, or c):\n"
                   "a. Read color from image\n"
                   "b. Read color from webcam\n"
                   "c. Take live picture from webcam\n"
                   "Your choice: ")

    if option.lower() == 'a':
        img_path = input("Enter the image path: ")
        img = color_detector.read_color_from_image(img_path)
        color_detector.detect_color(img)

    elif option.lower() == 'b':
        color_detector.read_color_from_webcam()

    elif option.lower() == 'c':
        color_detector.take_live_picture()

    else:
        print("Invalid option. Please choose either 'a', 'b', or 'c'.")
