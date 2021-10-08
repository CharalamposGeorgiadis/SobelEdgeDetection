import cv2
import numpy as np
from numba import jit


# This command is utilized by the Numba library in order to speed up the code
@jit(nopython=True)
# Function that performs sobel edge detection on a video frame
# param f: video frame
# param sX: sobel kernel that will be used to calculate approximations of the derivatives for horizontal changes
# param sX: sobel kernel that will be used to calculate approximations of the derivatives for vertical changes
# return: edge image
def mySobel(f, sX, sY):
    output = np.zeros(f.shape)
    for i in range(f.shape[0] - 2):
        for j in range(f.shape[1] - 2):
            # Initializing the result of the 2D convolution of each sobel kernel with the current window
            x = 0
            y = 0
            si = 0
            for k in range(i, i + 3):
                sj = 0
                for l in range(j, j + 3):
                    # Calculating the sum of the element-wise multiplication of each sobel kernel with the current
                    # window
                    x += sX[si][sj] * f[k][l]
                    y += sY[si][sj] * f[k][l]
                    sj += 1
                si += 1
            # Calculating the gradient magnitude of the current window
            output[i][j] = np.sqrt(x ** 2 + y ** 2)
    return output


# Initializing the two sobel kernels
sobelX = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
sobelY = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
# Reading the video
cap = cv2.VideoCapture('input.avi')
if cap.isOpened():
    # Defining the codec and creating a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 720), 0)
    sobel_image = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            # Converting each frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Performing sobel edge detection of every frame that is read
            sobel_image.append(mySobel(frame, sobelX, sobelY).astype(np.uint8))
            if cv2.waitKey(24) & 0xFF == ord('q'):
                break
        else:
            break
    # Displaying the video
    for frame in sobel_image:
        cv2.imshow('Sobel Video', frame)
        # Saving each frame of the sobel video
        out.write(frame)
        if cv2.waitKey(24) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
cap.release()
