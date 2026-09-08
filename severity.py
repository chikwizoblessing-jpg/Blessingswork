import cv2
import numpy as np

def estimate_severity(img_path):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define a color range for "unhealthy" leaf tissue
    # (brownish/yellowish tones — tweak these based on your dataset)
    lower_unhealthy = np.array([10, 40, 40])
    upper_unhealthy = np.array([35, 255, 255])
    
    # Define a color range for healthy green tissue
    lower_healthy = np.array([35, 40, 40])
    upper_healthy = np.array([85, 255, 255])
    
    unhealthy_mask = cv2.inRange(img, lower_unhealthy, upper_unhealthy)
    healthy_mask = cv2.inRange(img, lower_healthy, upper_healthy)
    
    unhealthy_pixels = np.sum(unhealthy_mask > 0)
    healthy_pixels = np.sum(healthy_mask > 0)
    total_leaf_pixels = unhealthy_pixels + healthy_pixels
    
    if total_leaf_pixels == 0:
        return 0
    
    severity_percent = (unhealthy_pixels / total_leaf_pixels) * 100
    return round(severity_percent, 1)

if __name__ == "__main__":
    severity = estimate_severity("test_leaf.jpg")
    print(f"Estimated affected leaf area: {severity}%")
    
    if severity < 10:
        urgency = "Low — monitor"
    elif severity < 30:
        urgency = "Medium — treat within the week"
    else:
        urgency = "High — treat immediately"
    print(f"Urgency: {urgency}")