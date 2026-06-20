import json


def print_json(data):
    """
    Pretty-print the entire JSON response.
    """
    print(json.dumps(data, indent=4))


def print_paths(data, prefix=""):
    """
    Recursively print all JSON paths.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            print(new_prefix)
            print_paths(value, new_prefix)

    elif isinstance(data, list) and len(data) > 0:
        print_paths(data[0], f"{prefix}[0]")


def get_paths(data, prefix=""):
    """
    Return a list of all JSON paths.
    """
    paths = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            paths.extend(get_paths(value, new_prefix))

    elif isinstance(data, list):
        if data:
            paths.extend(get_paths(data[0], f"{prefix}[0]"))

    else:
        paths.append(prefix)

    return paths


def show_features(data):
    """
    Display all available features in forecastday[0]['day'].
    """
    day_data = data["forecast"]["forecastday"][0]["day"]

    print("\nAvailable Daily Features:\n")

    for feature, value in sorted(day_data.items()):
        print(f"{feature}: {value}")


def get_features(data):
    """
    Return all available day features as a list.
    """
    day_data = data["forecast"]["forecastday"][0]["day"]

    return sorted(day_data.keys())


def categorize_features(data):
    """
    Group features by type.
    """
    day_data = data["forecast"]["forecastday"][0]["day"]

    numerical = []
    categorical = []
    nested = []

    for key, value in day_data.items():

        if isinstance(value, (int, float)):
            numerical.append(key)

        elif isinstance(value, str):
            categorical.append(key)

        elif isinstance(value, dict):
            nested.append(key)

    print("\nNumerical Features:")
    for item in sorted(numerical):
        print(f"  {item}")

    print("\nCategorical Features:")
    for item in sorted(categorical):
        print(f"  {item}")

    print("\nNested Features:")
    for item in sorted(nested):
        print(f"  {item}")


def build_record(day):
    """
    Automatically create a record from a forecast day.
    """
    record = {
        "date": day["date"],
        **day["day"]
    }

    return record