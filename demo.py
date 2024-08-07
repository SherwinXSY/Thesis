import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

#initialize global variable
MIN_CURVE_LENGTH = 300
MAX_CURVE_CHANGE = 2000

last_left_line = None
last_right_line = None
last_left_poly = None
last_right_poly = None
last_right_curvature = None
last_left_curvature = None


def capture_camera():
    cap = cv2.VideoCapture(0)  # default camera is 0
    if not cap.isOpened():
        print("Camera Not Available") 
        return None
    return cap

def area_of_interest(frame):
    mask = np.zeros_like(frame)
    vertices = np.array([[550, 260], [730, 260], [1280, 600], [0, 600]])
    cv2.fillPoly(mask, pts=[vertices], color=[255,255,255])
    #cv2.imshow("mask",mask)
    masked_frame = cv2.bitwise_and(frame,mask)
    
    
    return masked_frame

def preprocess_frame(frame):

    #image mask
    #print(frame.shape)
    #image size = 480*640
    frame_H = frame.shape[0]
    frame_W = frame.shape[1]
    #find_line(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imshow("Gaussian", blur)
    edges = cv2.Canny(blur, 50, 150)  

    edges_cropped = area_of_interest(edges)
    cv2.imshow("process", edges_cropped)
    return edges_cropped

def detect_lane_lines(edges):
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, threshold=15, minLineLength=15, maxLineGap=40)
    return lines

def classify_lines(lines, min_slope_threshold=0.5):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6) 
        
        # Remove horizontal and near-horizontal lines
        if abs(slope) < min_slope_threshold:
            continue
        
        if x1 < 640 and x2 < 640:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))
    
    return left_lines, right_lines

def average_lines(lines):
    x_coords = []
    y_coords = []
    
    for x1, y1, x2, y2 in lines:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    if len(x_coords) < 7 or len(y_coords) < 7:
        return None, None
    
    poly = np.polyfit(y_coords, x_coords, deg=2)
    
    y_min = min(y_coords)
    y_max = max(y_coords)
    y_fit = np.linspace(y_min, y_max, num=100)
    x_fit = np.polyval(poly, y_fit)
    fitted_line = [(int(x), int(y)) for x, y in zip(x_fit, y_fit)]
    
    #check length
    length = np.sqrt((x_fit[-1] - x_fit[0])**2 + (y_fit[-1] - y_fit[0])**2)
    #print(length)
    if length < MIN_CURVE_LENGTH:
        return None, None
    return fitted_line, poly

def draw_fitted_curve(frame, fitted_line, color=(0, 255, 0), thickness=3):
    if fitted_line is not None:
        for i in range(len(fitted_line) - 1):
            cv2.line(frame, fitted_line[i], fitted_line[i + 1], color, thickness)

def calculate_curvature(poly, y_eval):
    if poly is None:
        return None

    # Calculate the curvature of the polynomial at a given y position (y_eval)
    # poly is the coefficients of the polynomial
    A = poly[0]
    B = poly[1]
    
    # Calculate curvature using the formula for a 2nd degree polynomial
    curvature = ((1 + (2 * A * y_eval + B)**2)**1.5) / abs(2 * A)
    #print(curvature)
    
    return curvature


def calculate_steering_angle(lines, frame_width):
    if lines is None:
        return None

    #seperate left and right lines
    left_lines = [line for line in lines if line[0][0] < frame_width / 2]
    right_lines = [line for line in lines if line[0][0] >= frame_width / 2]
    
    if not left_lines or not right_lines:
        return None

    
    left_line = np.mean(left_lines, axis=0).astype(int)
    right_line = np.mean(right_lines, axis=0).astype(int)
    
    
    left_slope = (left_line[0][3] - left_line[0][1]) / (left_line[0][2] - left_line[0][0])
    right_slope = (right_line[0][3] - right_line[0][1]) / (right_line[0][2] - right_line[0][0])
    
    # averaging slope
    steering_angle = np.degrees(np.arctan((left_slope + right_slope) / 2))
    return steering_angle

def curvature_to_steering_angle(left_poly, right_poly, frame_width, y_eval):
    if left_poly is None or right_poly is None:
        return None
    
    # Calculate the desired path by averaging the left and right polynomials
    center_poly = (left_poly + right_poly) / 2
    
    # Calculate the vehicle's position relative to the center of the lane
    vehicle_center = frame_width / 2
    lane_center = np.polyval(center_poly, y_eval)
    deviation = vehicle_center - lane_center
    
    # Calculate the curvature of the desired path
    curvature = calculate_curvature(center_poly, y_eval)
    
    # Convert curvature to steering angle
    car_length = 2.5  
    if curvature != 0:
        steering_angle = np.arctan(deviation / car_length)
    else:
        steering_angle = 0
    
    steering_angle = np.degrees(steering_angle)
    return steering_angle


def display_video(cap):

    global last_left_line, last_right_line, last_left_poly, last_right_poly, last_left_curvature, last_right_curvature
    while cap.isOpened():
        ret, frame = cap.read() # load one frame
        if not ret:
            break

        
        edges = preprocess_frame(frame)
        lines = detect_lane_lines(edges)
        
        if lines is not None:
            left_lines, right_lines = classify_lines(lines)
        if left_lines is None or right_lines is None:
            print("No lanes detected in this frame, using previous lanes...")
            fitted_left_line = last_left_line
            fitted_right_line = last_right_line
            left_poly = last_left_poly
            right_poly = last_right_poly
        else:
            fitted_left_line, left_poly = average_lines(left_lines)
            fitted_right_line, right_poly = average_lines(right_lines)
            
            # Check if we got valid lines, otherwise use the previous ones
            if fitted_left_line is not None and left_poly is not None:
                left_curvature = calculate_curvature(left_poly, frame.shape[0])
                if last_left_curvature is not None and abs(left_curvature - last_left_curvature) > MAX_CURVE_CHANGE:
                    fitted_left_line = last_left_line
                    left_poly = last_left_poly
                else:
                    last_left_line = fitted_left_line
                    last_left_poly = left_poly
                    last_left_curvature = left_curvature
            else:
                fitted_left_line = last_left_line
                left_poly = last_left_poly

            if fitted_right_line is not None and right_poly is not None:
                right_curvature = calculate_curvature(right_poly, frame.shape[0])
                if last_right_curvature is not None and abs(right_curvature - last_right_curvature) > MAX_CURVE_CHANGE:
                    fitted_right_line = last_right_line
                    right_poly = last_right_poly
                else:
                    last_right_line = fitted_right_line
                    last_right_poly = right_poly
                    last_right_curvature = right_curvature
            else:
                fitted_right_line = last_right_line
                right_poly = last_right_poly
        
        draw_fitted_curve(frame, fitted_left_line, color=(0, 255, 0), thickness=5)
        draw_fitted_curve(frame, fitted_right_line, color=(0, 0, 255), thickness=5)
        print(left_poly)
        print(right_poly)
        # Calculate the average curvature and convert it to the steering angle
        if last_left_curvature is not None and last_right_curvature is not None:
            avg_curvature = (last_left_curvature + last_right_curvature) / 2
            steering_angle = curvature_to_steering_angle(left_poly, right_poly, frame.shape[1], frame.shape[0]) + 90
            if steering_angle is not None:
                cv2.putText(frame, f"Steering Angle: {steering_angle:.2f}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Lane detection", frame)
        #steering_angle = calculate_steering_angle(lines, frame.shape[1]) 
        
        
        
        # if steering_angle is not None:
        #     cv2.putText(frame, f"Steering Angle: {steering_angle:.2f}", (50, 50), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
        
        #     cv2.imshow("Lane Detection", frame) 
        # key = cv2.waitKey(0)
        cv2.waitKey(0) #uncomment this line to play video
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #cap = capture_video()  # capture from camera
    video_path = "G:\python\lane_detection\project_video6.mp4"
    cap = cv2.VideoCapture(video_path)
    if cap:
        display_video(cap)  
