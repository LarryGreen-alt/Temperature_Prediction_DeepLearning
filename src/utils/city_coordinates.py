'''
This is the list of cities and ajoined cooridnated.
This provides specific locations for the weather data to recall.
'''

CITY_COORDINATES = {
    # Texas
    "dallas": (32.7767, -96.7970),
    "houston": (29.7604, -95.3698),
    "austin": (30.2672, -97.7431),
    "san antonio": (29.4241, -98.4936),
    "fort worth": (32.7555, -97.3308),
    "el paso": (31.7619, -106.4850),
    "arlington": (32.7357, -97.1081),
    "corpus christi": (27.8006, -97.3964),
    "plano": (33.0198, -96.6989),
    "lubbock": (33.5779, -101.8552),
    "amarillo": (35.2219, -101.8313),
    "waco": (31.5493, -97.1467),
    "mcallen": (26.2034, -98.2300),

    # Northeast
    "new york": (40.7128, -74.0060),
    "boston": (42.3601, -71.0589),
    "philadelphia": (39.9526, -75.1652),
    "pittsburgh": (40.4406, -79.9959),
    "buffalo": (42.8864, -78.8784),
    "baltimore": (39.2904, -76.6122),
    "washington dc": (38.9072, -77.0369),
    "newark": (40.7357, -74.1724),
    "providence": (41.8240, -71.4128),

    # Southeast
    "miami": (25.7617, -80.1918),
    "orlando": (28.5383, -81.3792),
    "tampa": (27.9506, -82.4572),
    "jacksonville": (30.3322, -81.6557),
    "atlanta": (33.7490, -84.3880),
    "charlotte": (35.2271, -80.8431),
    "raleigh": (35.7796, -78.6382),
    "charleston": (32.7765, -79.9311),
    "savannah": (32.0809, -81.0912),
    "nashville": (36.1627, -86.7816),
    "memphis": (35.1495, -90.0490),
    "louisville": (38.2527, -85.7585),
    "new orleans": (29.9511, -90.0715),
    "birmingham": (33.5186, -86.8104),

    # Midwest
    "chicago": (41.8781, -87.6298),
    "detroit": (42.3314, -83.0458),
    "cleveland": (41.4993, -81.6944),
    "columbus": (39.9612, -82.9988),
    "cincinnati": (39.1031, -84.5120),
    "indianapolis": (39.7684, -86.1581),
    "milwaukee": (43.0389, -87.9065),
    "minneapolis": (44.9778, -93.2650),
    "st louis": (38.6270, -90.1994),
    "kansas city": (39.0997, -94.5786),
    "omaha": (41.2565, -95.9345),
    "des moines": (41.5868, -93.6250),

    # Mountain and Southwest
    "denver": (39.7392, -104.9903),
    "colorado springs": (38.8339, -104.8214),
    "salt lake city": (40.7608, -111.8910),
    "boise": (43.6150, -116.2023),
    "phoenix": (33.4484, -112.0740),
    "tucson": (32.2226, -110.9747),
    "albuquerque": (35.0844, -106.6504),
    "las vegas": (36.1699, -115.1398),
    "reno": (39.5296, -119.8138),

    # West Coast
    "los angeles": (34.0522, -118.2437),
    "san diego": (32.7157, -117.1611),
    "san francisco": (37.7749, -122.4194),
    "sacramento": (38.5816, -121.4944),
    "san jose": (37.3382, -121.8863),
    "fresno": (36.7378, -119.7871),
    "portland": (45.5152, -122.6784),
    "seattle": (47.6062, -122.3321),
    "spokane": (47.6588, -117.4260),

    # Alaska and Hawaii
    "anchorage": (61.2181, -149.9003),
    "fairbanks": (64.8378, -147.7164),
    "honolulu": (21.3099, -157.8581),

    # International cities
    "toronto": (43.6532, -79.3832),
    "vancouver": (49.2827, -123.1207),
    "mexico city": (19.4326, -99.1332),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "rome": (41.9028, 12.4964),
    "madrid": (40.4168, -3.7038),
    "tokyo": (35.6762, 139.6503),
    "seoul": (37.5665, 126.9780),
    "beijing": (39.9042, 116.4074),
    "sydney": (-33.8688, 151.2093),

    # North Korea — lol
    "random": (40.563017, 126.852766),
}

def get_coordinates(city_name):
    city_name = city_name.lower().strip()

    if city_name not in CITY_COORDINATES:
        raise ValueError(
            f"City '{city_name}' not found."
        )

    return CITY_COORDINATES[city_name]