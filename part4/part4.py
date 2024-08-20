import matplotlib.pyplot as plot
import mediapipe as mp
import numpy as np
import math
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(b,c):
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(b[1]-c[1], b[0]-c[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return round(angle - 90,2)

video_file1 = "6.mp4"
cap = cv2.VideoCapture(video_file1)
video_file2 = "7.mp4"
cap2 = cv2.VideoCapture(video_file2)

t = 0

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened() and cap2.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            knee1 = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angle_x
            angle_x = calculate_angle(knee1, ankle1)
                       
        except:
            pass        
        
        ret2, frame2 = cap2.read()
        
        # Recolor image to RGB
        image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        image2.flags.writeable = False
      
        # Make detection
        results2 = pose.process(image2)
    
        # Recolor back to BGR
        image2.flags.writeable = True
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        try:
            landmarks2 = results2.pose_landmarks.landmark
            
            # Get coordinates
            knee2 = [landmarks2[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks2[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle2 = [landmarks2[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks2[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angle_y
            angle_y = calculate_angle(knee2, ankle2)

                       
        except:
            pass

        #print("tetha_x = ", angle_x, "tetha_y = ", angle_y)
        #calculate Rotation Matrix
        Q = [[math.cos(angle_y), math.sin(angle_x)*math.sin(angle_y), math.cos(angle_x)*math.sin(angle_y)],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [-math.sin(angle_y), math.sin(angle_x)*math.cos(angle_y), math.cos(angle_x)*math.cos(angle_y)]]
        #print("Q = ", Q)
        
        #calculate e & phi
        tr = Q[0][0]+Q[1][1]+Q[2][2]
        V1 = 1/2*Q[2][1]-Q[1][2]
        V2 = 1/2*Q[0][2]-Q[2][0]
        V3 = 1/2*Q[1][0]-Q[0][1]
        vect = [[V1], [V2], [V3]]
        phi = math.acos((tr-1)/2)
        E1 = 1/2*V1/math.sin(phi)
        E2 = 1/2*V2/math.sin(phi)
        E3 = 1/2*V3/math.sin(phi)
        e = [[E1], [E2], [E3]]
        #print("phi = ", phi, "e = ", e)

        #calculate quaternion
        r0 = math.cos(phi/2)
        R1 = math.sin(phi/2)*E1
        R2 = math.sin(phi/2)*E2
        R3 = math.sin(phi/2)*E3
        r = [[R1], [R2], [R3]]
        #print("r0 = ", r0, "r = ", r)

        #plot tetha_x 
        try:
            t += 1
            amp_1 = angle_x
            plot.subplot(2, 2, 1)
            plot.scatter(t, amp_1)
            plot.pause(0.0005)
        except:
            pass

        f = open("data_x.txt", "a")
        f.write(str(amp_1) + "\n")
        f.close()

        #plot tetha_y
        try:

            amp_2 = angle_y
            plot.subplot(2, 2, 2)
            plot.scatter(t, amp_2)
            plot.pause(0.0005)
        except:
            pass

        f = open("data_y.txt", "a")
        f.write(str(amp_2) + "\n")
        f.close()
        
        #plot quaternion
        try:
            amp_3 = r0
            plot.subplot(2, 2, 3)
            plot.scatter(t, amp_3)
            plot.pause(0.0005)
        except:
            pass

        f = open("data_r0.txt", "a")
        f.write(str(amp_3) + "\n")
        f.close()
    


        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )         
        mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 ) 
        cv2.imshow('Mediapipe Feed', image)
        cv2.imshow('Mediapipe Feed', image2)
        

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    #cap2.release()
    cv2.destroyAllWindows()
    plot.show()
