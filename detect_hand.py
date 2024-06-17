import cv2
import numpy as np

def detect_hand(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    
    # Thresholding to create binary image
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug: Display thresholded image
    cv2.imshow('Thresholded Image', thresh)
    
    # Filter contours to find potential hands
    potential_hands = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter based on area
        if area < 1000 or area > 10000:  # Adjust area threshold as needed
            continue
        
        # Calculate convex hull to estimate convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Ensure hull_area is not zero
        if hull_area == 0:
            continue
        
        solidity = float(area) / hull_area
        
        # Filter based on solidity (convexity)
        if solidity < 0.8:  # Adjust solidity threshold as needed
            continue
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Adjust aspect ratio threshold as needed
            continue
        
        # If all criteria are met, it's a potential hand
        potential_hands.append(contour)
        
        # Draw bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate the convexity defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        
        if defects is not None:
            # Count the number of fingers
            finger_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Calculate the length of all sides of the triangle
                a = np.linalg.norm(np.array(start) - np.array(end))
                b = np.linalg.norm(np.array(start) - np.array(far))
                c = np.linalg.norm(np.array(end) - np.array(far))
                
                # Apply the cosine rule to find the angle
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                
                # Ignore angles > 90 and ignore points too close to convex hull
                if angle <= np.pi / 2 and d > 10000:  # Angle less than 90 degrees and significant depth
                    finger_count += 1
                    cv2.circle(frame, far, 8, [211, 84, 0], -1)
            
            # Number of fingers is number of defects + 1 (for the thumb)
            finger_count += 1
            
            # Display the number of fingers
            cv2.putText(frame, f"Fingers: {finger_count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Debug: Display processed frame with bounding boxes
    cv2.imshow('Processed Frame', frame)
    
    return frame
