import imutils  ## Import imutils library for convenient image resizing operations
import cv2  ## Import OpenCV library for computer vision and image processing tasks

## Define the lower HSV (Hue, Saturation, Value) threshold for color detection
## HSV is better than RGB for color detection as it separates color from brightness
redLower = (58, 95, 80)

## Define the upper HSV threshold for color detection
## Together with redLower, this creates a range to detect specific colors (despite variable name, this detects green/cyan colors)
redUpper = (147, 255, 255)

## Initialize video capture from camera
## 0 = default camera, 1 = external/second camera
camera = cv2.VideoCapture(1)

## Start infinite loop to continuously process video frames
while True:

    ## Read a single frame from the camera
    ## grabbed = boolean (True if frame was successfully read)
    ## frame = the actual image/frame captured
    (grabbed, frame) = camera.read()

    ## Resize the frame to width of 1000 pixels (maintains aspect ratio)
    ## This makes processing faster and more consistent
    frame = imutils.resize(frame, width=1000)
    
    ## Apply Gaussian blur to reduce noise and improve color detection
    ## (11, 11) is the kernel size - larger values = more blur
    ## 0 is the standard deviation (auto-calculated when set to 0)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    
    ## Convert the blurred image from BGR (OpenCV's default) to HSV color space
    ## HSV makes it easier to detect colors under varying lighting conditions
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    ## Create a binary mask that isolates pixels within the specified HSV range
    ## Pixels in range = white (255), pixels outside range = black (0)
    mask = cv2.inRange(hsv, redLower, redUpper)
    
    ## Erode the mask to remove small white noise/spots
    ## iterations=2 means apply erosion twice for stronger effect
    mask = cv2.erode(mask, None, iterations=2)
    
    ## Dilate the mask to restore the size of the detected object
    ## This fills in small holes and smooths the edges
    mask = cv2.dilate(mask, None, iterations=2)

    ## Find contours (outlines) of white regions in the mask
    ## cv2.RETR_EXTERNAL = only retrieve outer contours
    ## cv2.CHAIN_APPROX_SIMPLE = compress contours to save memory
    ## [-2] gets the contours list (compatibility with different OpenCV versions)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    ## Initialize center variable to None (will store object's center point)
    center = None
    
    ## Check if at least one contour was found
    if len(cnts) > 0:
        ## Find the largest contour by area (assumes it's the object we want to track)
        c = max(cnts, key=cv2.contourArea)
        
        ## Get the minimum enclosing circle around the contour
        ## Returns center coordinates (x, y) and radius
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        ## Calculate moments to find the centroid (center of mass) of the contour
        M = cv2.moments(c)
        
        ## Calculate the actual center point using moments
        ## m10/m00 = x coordinate, m01/m00 = y coordinate
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        ## Only process if the object is large enough (radius > 10 pixels)
        ## This filters out small noise that passed through erosion
        if radius > 10:
            ## Draw a yellow circle around the detected object
            ## (int(x), int(y)) = center, int(radius) = radius
            ## (0, 255, 255) = yellow color in BGR, 2 = line thickness
            cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
            
            ## Draw a small red dot at the center of the object
            ## 5 = radius of dot, (0, 0, 255) = red in BGR, -1 = filled circle
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            ## Check if object is very close (radius > 250 pixels)
            if radius > 250:
                print("stop")  ## Object is too close, print "stop"
            else:
                ## Determine object position and print directional commands
                
                ## If object is on the left side of frame (x < 150)
                if(center[0] < 150):
                    print("Right")  ## Tell to move right to center it
                
                ## If object is on the right side of frame (x > 450)
                elif(center[0] > 450):
                    print("Left")  ## Tell to move left to center it
                
                ## If object is too far (radius < 250)
                elif(radius < 250):
                    print("Front")  ## Tell to move forward/closer
                
                ## Otherwise object is well-positioned
                else:
                    print("Stop")  ## Object is in good position
    
    ## Display the processed frame in a window named "Frame"
    cv2.imshow("Frame", frame)
    
    ## Wait 1 millisecond for a key press
    ## 0xFF masks the key to get only the last 8 bits (for compatibility)
    key = cv2.waitKey(1) & 0xFF
    
    ## If 'q' key is pressed, break out of the loop
    if key == ord("q"):
        break

## Release the camera resource (important for cleanup)
camera.release()

## Close all OpenCV windows
cv2.destroyAllWindows()
