from typing import List, Dict, Any

VEHICLE_WEIGHTS = {
    "pedestrian": 3.0,
    "bicycle": 2.5,
    "motorcycle": 2.5,
    "car": 1.0,
    "bus": 2.2,
    "truck": 2.5
}

def compute_severity(
    detected_vehicles: List[Dict[str, Any]],
    overlap_ratio: float = 0.0,
    speed_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluates vehicle types, collision overlap, and mass weights 
    to score accident severity (Minor, Moderate, Critical).
    """
    vehicle_types = [v.get("class", "car").lower() for v in detected_vehicles]
    total_vehicles = len(vehicle_types)

    if total_vehicles < 2 and overlap_ratio == 0:
        return {
            "severity_level": "None",
            "score": 0.0,
            "triage_priority": "Low",
            "dispatched_units": []
        }

    # Base score calculated from involved vehicle weights
    weight_score = sum(VEHICLE_WEIGHTS.get(v, 1.0) for v in vehicle_types)

    # Composite severity score: weight score + overlap ratio boost + vehicle count factor
    severity_score = (weight_score * 12.0) + (overlap_ratio * 30.0) + (total_vehicles * 5.0) * speed_factor

    if severity_score >= 55.0 or "pedestrian" in vehicle_types or "motorcycle" in vehicle_types:
        severity_level = "Critical"
        triage_priority = "P1 - Immediate (Code 3)"
        dispatched_units = ["Advanced Life Support Ambulance", "Police Highway Patrol", "Heavy Rescue Fire Unit"]
    elif severity_score >= 35.0:
        severity_level = "Moderate"
        triage_priority = "P2 - Urgent"
        dispatched_units = ["Basic Life Support Ambulance", "Traffic Police Unit"]
    else:
        severity_level = "Minor"
        triage_priority = "P3 - Standard"
        dispatched_units = ["Traffic Police Patrol"]

    return {
        "severity_level": severity_level,
        "score": round(severity_score, 2),
        "triage_priority": triage_priority,
        "vehicle_breakdown": {v: vehicle_types.count(v) for v in set(vehicle_types)},
        "dispatched_units": dispatched_units
    }
