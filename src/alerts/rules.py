
def generate_alerts(ndvi_change, ndmi_change,
                    ndvi_threshold=-0.10,
                    ndmi_threshold=-0.10):
    """Flag significant vegetation-index declines."""
    alerts = []

    if ndvi_change <= ndvi_threshold:
        alerts.append("Vegetation greenness declined")

    if ndmi_change <= ndmi_threshold:
        alerts.append("Vegetation moisture indicator declined")

    if not alerts:
        alerts.append("No significant index decline detected")

    return alerts
