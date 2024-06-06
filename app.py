import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os

# Initialize deque arrays to handle color points for both hands
bpoints1, bpoints2 = [deque(maxlen=1024)], [deque(maxlen=1024)]
gpoints1, gpoints2 = [deque(maxlen=1024)], [deque(maxlen=1024)]
rpoints1, rpoints2 = [deque(maxlen=1024)], [deque(maxlen=1024)]
ypoints1, ypoints2 = [deque(maxlen=1024)], [deque(maxlen=1024)]
ppoints1, ppoints2 = [deque(maxlen=1024)], [deque(maxlen=1024)]

# Indexes for points in particular arrays of specific color for both hands
blue_index1, blue_index2 = 0, 0
green_index1, green_index2 = 0, 0
red_index1, red_index2 = 0, 0
yellow_index1, yellow_index2 = 0, 0
pink_index1, pink_index2 = 0, 0

# Kernel for dilation purpose
kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 192, 203)]
colorIndex1 = 0
colorIndex2 = 0

# Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (615, 1), (710, 65), (255, 192, 203), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "PINK", (655, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Undo stack and brush size variable
undo_stack = []
brush_size = 5

ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw buttons on frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
    frame = cv2.rectangle(frame, (615, 1), (710, 65), (255, 192, 203), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "PINK", (655, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Post process the result
    if result.multi_hand_landmarks:
        for hand_no, handslms in enumerate(result.multi_hand_landmarks):
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

            if (thumb[1] - center[1] < 30):
                # Save the current state before starting a new stroke
                if hand_no == 0:
                    undo_stack.append((bpoints1.copy(), gpoints1.copy(), rpoints1.copy(), ypoints1.copy(), ppoints1.copy(),
                                       blue_index1, green_index1, red_index1, yellow_index1, pink_index1))

                    bpoints1.append(deque(maxlen=512))
                    blue_index1 += 1
                    gpoints1.append(deque(maxlen=512))
                    green_index1 += 1
                    rpoints1.append(deque(maxlen=512))
                    red_index1 += 1
                    ypoints1.append(deque(maxlen=512))
                    yellow_index1 += 1
                    ppoints1.append(deque(maxlen=512))
                    pink_index1 += 1
                else:
                    undo_stack.append((bpoints2.copy(), gpoints2.copy(), rpoints2.copy(), ypoints2.copy(), ppoints2.copy(),
                                       blue_index2, green_index2, red_index2, yellow_index2, pink_index2))

                    bpoints2.append(deque(maxlen=512))
                    blue_index2 += 1
                    gpoints2.append(deque(maxlen=512))
                    green_index2 += 1
                    rpoints2.append(deque(maxlen=512))
                    red_index2 += 1
                    ypoints2.append(deque(maxlen=512))
                    yellow_index2 += 1
                    ppoints2.append(deque(maxlen=512))
                    pink_index2 += 1

            if center[1] <= 65:
                if 40 <= center[0] <= 140:
                    bpoints1, gpoints1, rpoints1, ypoints1, ppoints1 = [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)]
                    blue_index1, green_index1, red_index1, yellow_index1, pink_index1 = 0, 0, 0, 0, 0

                    bpoints2, gpoints2, rpoints2, ypoints2, ppoints2 = [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)]
                    blue_index2, green_index2, red_index2, yellow_index2, pink_index2 = 0, 0, 0, 0, 0
                    paintWindow[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    if hand_no == 0:
                        colorIndex1 = 0  # Blue
                    else:
                        colorIndex2 = 0  # Blue
                elif 275 <= center[0] <= 370:
                    if hand_no == 0:
                        colorIndex1 = 1  # Green
                    else:
                        colorIndex2 = 1  # Green
                elif 390 <= center[0] <= 485:
                    if hand_no == 0:
                        colorIndex1 = 2  # Red
                    else:
                        colorIndex2 = 2  # Red
                elif 505 <= center[0] <= 600:
                    if hand_no == 0:
                        colorIndex1 = 3  # Yellow
                    else:
                        colorIndex2 = 3  # Yellow
                elif 615 <= center[0] <= 710:
                    if hand_no == 0:
                        colorIndex1 = 4  # Pink
                    else:
                        colorIndex2 = 4  # Pink
            else:
                if hand_no == 0:
                    if colorIndex1 == 0:
                        bpoints1[blue_index1].appendleft(center)
                    elif colorIndex1 == 1:
                        gpoints1[green_index1].appendleft(center)
                    elif colorIndex1 == 2:
                        rpoints1[red_index1].appendleft(center)
                    elif colorIndex1 == 3:
                        ypoints1[yellow_index1].appendleft(center)
                    elif colorIndex1 == 4:
                        ppoints1[pink_index1].appendleft(center)
                else:
                    if colorIndex2 == 0:
                        bpoints2[blue_index2].appendleft(center)
                    elif colorIndex2 == 1:
                        gpoints2[green_index2].appendleft(center)
                    elif colorIndex2 == 2:
                        rpoints2[red_index2].appendleft(center)
                    elif colorIndex2 == 3:
                        ypoints2[yellow_index2].appendleft(center)
                    elif colorIndex2 == 4:
                        ppoints2[pink_index2].appendleft(center)
    else:
        hand1_present = False
        hand2_present = False

    # Draw lines of all the colors on the canvas and frame
    points1 = [bpoints1, gpoints1, rpoints1, ypoints1, ppoints1]
    points2 = [bpoints2, gpoints2, rpoints2, ypoints2, ppoints2]

    for i in range(len(points1)):
        for j in range(len(points1[i])):
            for k in range(1, len(points1[i][j])):
                if points1[i][j][k - 1] is None or points1[i][j][k] is None:
                    continue
                cv2.line(frame, points1[i][j][k - 1], points1[i][j][k], colors[i], brush_size)
                cv2.line(paintWindow, points1[i][j][k - 1], points1[i][j][k], colors[i], brush_size)

    for i in range(len(points2)):
        for j in range(len(points2[i])):
            for k in range(1, len(points2[i][j])):
                if points2[i][j][k - 1] is None or points2[i][j][k] is None:
                    continue
                cv2.line(frame, points2[i][j][k - 1], points2[i][j][k], colors[i], brush_size)
                cv2.line(paintWindow, points2[i][j][k - 1], points2[i][j][k], colors[i], brush_size)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)
    key = cv2.waitKey(1)

    if key == ord('s') or key == ord('S'):
        desktop_path = "C:/Users/netware/Desktop/Datstr/"
        if not os.path.exists(desktop_path):
            os.makedirs(desktop_path)
        file_number = 1
        while os.path.exists(desktop_path + f"{file_number}.png"):
            file_number += 1
        filename = desktop_path + f"{file_number}.png"
        cv2.imwrite(filename, paintWindow)
        print(f"Canvas drawing saved as '{filename}'")
    elif key == ord('q') or key == ord('Q'):
        break
    elif key == ord('u') or key == ord('U'):
        if undo_stack:
            bpoints1, gpoints1, rpoints1, ypoints1, ppoints1, blue_index1, green_index1, red_index1, yellow_index1, pink_index1 = undo_stack.pop()
            bpoints2, gpoints2, rpoints2, ypoints2, ppoints2, blue_index2, green_index2, red_index2, yellow_index2, pink_index2 = undo_stack.pop()
            paintWindow[67:, :, :] = 255  # Clear the canvas area
            # Redraw the canvas based on the current points
            for i in range(len(points1)):
                for j in range(len(points1[i])):
                    for k in range(1, len(points1[i][j])):
                        if points1[i][j][k - 1] is None or points1[i][j][k] is None:
                            continue
                        cv2.line(paintWindow, points1[i][j][k - 1], points1[i][j][k], colors[i], brush_size)
            for i in range(len(points2)):
                for j in range(len(points2[i])):
                    for k in range(1, len(points2[i][j])):
                        if points2[i][j][k - 1] is None or points2[i][j][k] is None:
                            continue
                        cv2.line(paintWindow, points2[i][j][k - 1], points2[i][j][k], colors[i], brush_size)
    elif key == ord('+'):
        brush_size = min(20, brush_size + 1)  # Cap the brush size to 20
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)  # Minimum brush size is 1

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
