"""
Traffic Calculator Service
Calculates traffic levels and congestion scores
"""

from app.core.config import settings


def calculate_traffic_level(vehicles: int, pedestrians: int) -> str:
    """
    Calculate traffic level based on vehicle and pedestrian counts.
    
    Args:
        vehicles: Number of vehicles detected
        pedestrians: Number of pedestrians detected
        
    Returns:
        Traffic level: 'low', 'medium', or 'high'
    """
    if vehicles > settings.TRAFFIC_HIGH_VEHICLES or pedestrians > settings.TRAFFIC_HIGH_PEDESTRIANS:
        return "high"
    elif vehicles > settings.TRAFFIC_MEDIUM_VEHICLES or pedestrians > settings.TRAFFIC_MEDIUM_PEDESTRIANS:
        return "medium"
    else:
        return "low"


def calculate_congestion_score(vehicles: int, pedestrians: int, cyclists: int) -> float:
    """
    Calculate a congestion score from 0-100.
    
    The score considers:
    - Vehicle count (weighted heavily)
    - Pedestrian count (moderate weight)
    - Cyclist count (lower weight)
    
    Args:
        vehicles: Number of vehicles detected
        pedestrians: Number of pedestrians detected
        cyclists: Number of cyclists detected
        
    Returns:
        Congestion score from 0 to 100
    """
    # Weights for different traffic types
    vehicle_weight = 2.0
    pedestrian_weight = 0.5
    cyclist_weight = 0.3
    
    # Maximum expected values for normalisation
    max_vehicles = 100
    max_pedestrians = 300
    max_cyclists = 50
    
    # Calculate normalised scores
    vehicle_score = min(vehicles / max_vehicles, 1.0) * vehicle_weight
    pedestrian_score = min(pedestrians / max_pedestrians, 1.0) * pedestrian_weight
    cyclist_score = min(cyclists / max_cyclists, 1.0) * cyclist_weight
    
    # Combined score (normalised to 0-100)
    total_weight = vehicle_weight + pedestrian_weight + cyclist_weight
    congestion_score = ((vehicle_score + pedestrian_score + cyclist_score) / total_weight) * 100
    
    return round(min(congestion_score, 100.0), 2)


def get_traffic_status_message(traffic_level: str, vehicles: int, pedestrians: int) -> str:
    """
    Generate a human-readable traffic status message.
    
    Args:
        traffic_level: Current traffic level
        vehicles: Number of vehicles
        pedestrians: Number of pedestrians
        
    Returns:
        Descriptive status message
    """
    if traffic_level == "high":
        return f"Heavy traffic: {vehicles} vehicles, {pedestrians} pedestrians. Expect delays."
    elif traffic_level == "medium":
        return f"Moderate traffic: {vehicles} vehicles, {pedestrians} pedestrians. Normal conditions."
    else:
        return f"Light traffic: {vehicles} vehicles, {pedestrians} pedestrians. Clear conditions."


def estimate_travel_impact(traffic_level: str) -> dict:
    """
    Estimate the impact on travel times based on traffic level.
    
    Args:
        traffic_level: Current traffic level
        
    Returns:
        Dictionary with travel time multiplier and description
    """
    impacts = {
        "low": {
            "multiplier": 1.0,
            "delay_minutes": 0,
            "description": "No significant delays expected"
        },
        "medium": {
            "multiplier": 1.2,
            "delay_minutes": 5,
            "description": "Minor delays possible, add 5 minutes to journey"
        },
        "high": {
            "multiplier": 1.5,
            "delay_minutes": 15,
            "description": "Significant delays expected, add 15+ minutes to journey"
        }
    }
    
    return impacts.get(traffic_level, impacts["low"])
