def calculate_severity(vitals):
    """
    Rule-based severity scoring function.
    Takes a dictionary of patient vitals and returns an integer severity score.
    """
    score = 0

    # Heart Rate scoring
    hr = vitals.get('HR')
    if hr is not None:
        if hr < 50 or hr > 120:
            score += 2

    # Oxygen Saturation (SpO2) scoring
    spo2 = vitals.get('SpO2')
    if spo2 is not None and spo2 < 92:
        score += 3

    # Systolic Blood Pressure scoring
    bp_sys = vitals.get('BP_SYS')
    if bp_sys is not None and (bp_sys < 90 or bp_sys > 150):
        score += 2

    return score


def prioritize_alerts(alert_list):
    """
    Takes a list of alerts, attaches severity scores, and returns them sorted.
    Each alert should have a 'vitals' dictionary.
    """
    for alert in alert_list:
        alert['severity_score'] = calculate_severity(alert.get('vitals', {}))
    
    return sorted(alert_list, key=lambda x: x['severity_score'], reverse=True)


