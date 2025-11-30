# memory_bank.py
import json, os

MEM_FILE = "data/memory_bank.json"

def remember(key, value):
    os.makedirs(os.path.dirname(MEM_FILE), exist_ok=True)
    if os.path.exists(MEM_FILE):
        with open(MEM_FILE,"r",encoding="utf-8") as f:
            db = json.load(f)
    else:
        db = {}
    db[key] = value
    with open(MEM_FILE,"w",encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def recall(key):
    if not os.path.exists(MEM_FILE):
        return None
    with open(MEM_FILE,"r",encoding="utf-8") as f:
        db = json.load(f)
    return db.get(key)
