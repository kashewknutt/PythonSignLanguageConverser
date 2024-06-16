import cv2

def detect_hand(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7,7), 2)
    
    # Thresholding to create binary image
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug: Display thresholded image
    cv2.imshow('Thresholded Image', thresh)
    
    # Filter contours to find potential hands
    potential_hands = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter based on area
        if area < 10000 or area > 100000:  # Adjust area threshold as needed
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
        
        if aspect_ratio < 0.5:  # Adjust aspect ratio threshold as needed
            continue
        
        # Draw bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # If all criteria are met, it's a potential hand
        potential_hands.append((x, y, w, h))
    
    # Debug: Display processed frame with bounding boxes
    cv2.imshow('Processed Frame', frame)
    
    return frame
