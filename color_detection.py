import cv2
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

img_path = "colorpic.jpg"
img = cv2.imread(img_path)

index = ["color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

# Extract features (RGB values) and labels (color names) from the CSV data
features = csv[['R', 'G', 'B']].values
labels = csv['color_name'].values

# Create and train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(features, labels)

clicked = False
r = g = b = xpos = ypos = 0

def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

cv2.namedWindow('color detection')
cv2.setMouseCallback('color detection', draw_function)

while True:
    cv2.imshow("color detection", img)
    
    if clicked:
        # Predict the color name using KNN based on the RGB values
        test_data = np.array([[r, g, b]])
        color_name = knn.predict(test_data)[0]
        
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)
        color_info = f"{color_name} R={r} G={g} B={b}"
        
        # Display the color information on the image
        cv2.putText(img, color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # For very light colors, display text in black for better visibility
        if r + g + b >= 600:
            cv2.putText(img, color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
        clicked = False
    
    # Exit the loop if the 'Esc' key is pressed
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
