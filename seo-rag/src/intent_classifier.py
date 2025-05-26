import json

def load_patterns(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def classify_intent(query, patterns):
    query_l = query.lower()
    for intent, keywords in patterns.items():
        for k in keywords:
            if k in query_l:
                return intent
    return "unknown"

def load_groups(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def map_group(query, groups):
    query_l = query.lower()
    for group, keywords in groups.items():
        for k in keywords:
            if k in query_l:
                return group
    return "Misc Treatments"
