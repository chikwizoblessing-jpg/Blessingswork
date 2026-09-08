# A simple lookup table — expand this as you learn more about each disease
DISEASE_INFO = {
    "Tomato___Early_blight": {
        "favorable_humidity": (70, 100),  # % range where disease spreads fastest
        "favorable_temp_c": (24, 29),
        "treatment": "Remove affected leaves. Apply copper-based fungicide if available.",
    },
    "Tomato___Late_blight": {
        "favorable_humidity": (90, 100),
        "favorable_temp_c": (10, 24),
        "treatment": "Isolate affected plants immediately. Apply fungicide; late blight spreads fast.",
    },
    "Tomato___healthy": {
        "favorable_humidity": (0, 100),
        "favorable_temp_c": (18, 27),
        "treatment": "No action needed — maintain current care routine.",
    },
}

def get_recommendation(disease_class, current_temp_c, current_humidity_pct):
    info = DISEASE_INFO.get(disease_class)
    if info is None:
        return "No data available for this class yet."
    
    temp_lo, temp_hi = info["favorable_temp_c"]
    hum_lo, hum_hi = info["favorable_humidity"]
    conditions_favor_disease = (
        temp_lo <= current_temp_c <= temp_hi and
        hum_lo <= current_humidity_pct <= hum_hi
    )
    
    report = [f"Diagnosis: {disease_class}"]
    report.append(f"Treatment: {info['treatment']}")
    
    if "healthy" in disease_class.lower():
        report.append("Current conditions: not applicable (plant is healthy).")
    elif conditions_favor_disease:
        report.append(
            f"⚠️ Current conditions (temp {current_temp_c}°C, humidity {current_humidity_pct}%) "
            f"FAVOR disease spread. Act now."
        )
    else:
        report.append(
            f"Current conditions (temp {current_temp_c}°C, humidity {current_humidity_pct}%) "
            f"are less favorable for spread — still treat, but urgency is lower."
        )
    
    return "\n".join(report)

if __name__ == "__main__":
    # Example: values you'd normally get from predict.py and your sensors
    example_disease = "Tomato___Early_blight"
    example_temp = 26
    example_humidity = 82
    print(get_recommendation(example_disease, example_temp, example_humidity))
