"""Gin-configurable flight platform presets."""
import gin

@gin.configurable
def isr_orbit_preset():
    return {"platform": "isr_orbit", "altitude_range": (100, 500), "speed_range": (15, 40)}

@gin.configurable
def isr_plane_preset():
    return {"platform": "isr_plane", "altitude_range": (500, 5000), "speed_range": (50, 150)}

@gin.configurable
def fpv_racing_preset():
    return {"platform": "fpv_racing", "altitude_range": (5, 50), "speed_range": (30, 80)}

@gin.configurable
def ugv_wheeled_preset():
    return {"platform": "ugv_wheeled", "altitude_range": (0.5, 2), "speed_range": (2, 15)}

@gin.configurable
def satellite_preset():
    return {"platform": "satellite", "altitude_range": (200000, 500000), "speed_range": (7000, 7000)}
