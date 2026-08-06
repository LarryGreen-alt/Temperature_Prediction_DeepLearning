'''
This is the list of cities and ajoined cooridnated.
This provides specific locations for the weather data to recall.
'''

CITY_COORDINATES = {
    "dallas": (32.7767, -96.7970),
    "houston": (29.7604, -95.3698),
    "austin": (30.2672, -97.7431),
    "san antonio": (29.4241, -98.4936),
    "fort worth": (32.7555, -97.3308),

    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "miami": (25.7617, -80.1918),
    "seattle": (47.6062, -122.3321),

    "denver": (39.7392, -104.9903),
    "phoenix": (33.4484, -112.0740),
    "las vegas": (36.1699, -115.1398),
    "atlanta": (33.7490, -84.3880),
    "boston": (42.3601, -71.0589),

    #North Korea -lol
    "random": (40.563017, 126.852766)
}

def get_coordinates(city_name):
    city_name = city_name.lower().strip()

    if city_name not in CITY_COORDINATES:
        raise ValueError(
            f"City '{city_name}' not found."
        )

    return CITY_COORDINATES[city_name]
