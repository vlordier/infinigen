import gin
import numpy as np

@gin.configurable
def heuristic_temperature(material_category: str, base_temp: float = 293.15, noise_scale: float = 5.0) -> float:
    category_temps = {
        "metal": base_temp + 15.0,
        "concrete": base_temp + 8.0,
        "asphalt": base_temp + 20.0,
        "vegetation": base_temp - 3.0,
        "water": base_temp - 5.0,
        "soil": base_temp + 2.0,
        "wood": base_temp + 5.0,
    }
    temp = category_temps.get(material_category, base_temp)
    temp += np.random.normal(0, noise_scale)
    return float(max(200.0, min(400.0, temp)))
