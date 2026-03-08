from app.core.config import settings


def calculate_traffic_level(vehicles: int, pedestrians: int) -> str:
    if vehicles > settings.TRAFFIC_HIGH_VEHICLES or pedestrians > settings.TRAFFIC_HIGH_PEDESTRIANS:
        return "high"
    elif vehicles > settings.TRAFFIC_MEDIUM_VEHICLES or pedestrians > settings.TRAFFIC_MEDIUM_PEDESTRIANS:
        return "medium"
    else:
        return "low"


def calculate_congestion_score(vehicles: int, pedestrians: int, cyclists: int) -> float:
    vehicle_score = min(vehicles / 100, 1.0) * 2.0
    pedestrian_score = min(pedestrians / 300, 1.0) * 0.5
    cyclist_score = min(cyclists / 50, 1.0) * 0.3

    total_weight = 2.0 + 0.5 + 0.3
    score = ((vehicle_score + pedestrian_score + cyclist_score) / total_weight) * 100

    return round(min(score, 100.0), 2)