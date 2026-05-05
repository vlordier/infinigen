import gin
from enum import Enum

class DamageProgressionStage(Enum):
    STAGE_0_INTACT = 0
    STAGE_1_MILD = 1
    STAGE_2_MODERATE = 2
    STAGE_3_SEVERE = 3
    STAGE_4_TOTAL = 4

NATURE_EARTHQUAKE_SEVERITY = {
    0: "Intact forest/terrain",
    1: "5-10% tree toppling, hairline ground cracks, minor rockfall",
    2: "20-40% trees down, landslides on steep slopes, root-plate craters, blocked streams",
    3: "50-70% forest flattened, major landslides burying roads, debris dams in rivers",
    4: "Complete forest destruction, terrain unrecognizable, all water courses blocked",
}

NATURE_WAR_SEVERITY = {
    0: "Intact forest/terrain",
    1: "1-3 small craters in clearings, peripheral scorching on tree edges",
    2: "Cluster of 5-10 medium craters, scorched treeline, burnt clearings, smoke columns",
    3: "Extensive crater fields, 40-60% canopy scorched/burnt, smouldering hot spots",
    4: "Total burn scar, overlapping craters, zero living vegetation, continuous smoke",
}

URBAN_EARTHQUAKE_SEVERITY = {
    0: "No damage",
    1: "Hairline cracks in walls, shifted furniture, 5-10% trees toppled",
    2: "Partial wall collapse (20-40% walls), moderate rubble",
    3: "Major structural failure (60-80% walls), extensive rubble",
    4: "Building reduced to rubble pile, roads destroyed",
}

URBAN_WAR_SEVERITY = {
    0: "No damage", 
    1: "Small craters (<5m), minor facade chips",
    2: "Medium craters (5-15m), facade holes, moderate scorching",
    3: "Large craters (>15m), collapsed facades, heavy scorching",
    4: "Near-total destruction, overlapping craters, burned-out landscape",
}

@gin.configurable
def get_progression_descriptions(damage_type="earthquake", is_urban=True):
    if is_urban:
        return URBAN_EARTHQUAKE_SEVERITY if damage_type == "earthquake" else URBAN_WAR_SEVERITY
    return NATURE_EARTHQUAKE_SEVERITY if damage_type == "earthquake" else NATURE_WAR_SEVERITY
