MECHANIC_PATTERNS = {
    "damage": [r"deal.*damage", r"damage", r"destroy"],
    "control": [r"capture", r"return.*to.*hand", r"discard"],
    "resource": [r"resource", r"generate", r"gain"],
    "defense": [r"shield", r"protect", r"defend"],
    "combat": [r"attack", r"combat", r"fight"],
    "support": [r"draw.*card", r"search", r"reveal"]
}

ASPECT_COLORS = {
    "Command": "Green",
    "Cunning": "Yellow",
    "Aggression": "Red",
    "Vigilance": "Blue"
}

CARD_NAME_VARIATIONS = {
    'sergent': 'sergeant',
    'sargent': 'sergeant',
    'wreker': 'wrecker',
    'maruader': 'marauder'
}