def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Look at the field in the image
    fields = img.find("field")
    if len(fields) == 0 and not img.exists("field"):
        return {"verdict": "INVALID", "failed_step": 1, "error": "No field detected in the image", "steps": trace}
    trace.append({"step": 1, "claim": "field exists in the image", "passed": True})

    # Step 2: Identify presence of animal in the field
    animals = img.find("animal")
    if len(animals) == 0:
        # Fallback to specific animal search if generic "animal" detection fails
        animals = img.find("horse")
    if len(animals) == 0:
        return {"verdict": "INVALID", "failed_step": 2, "error": "No animal found in the field", "steps": trace}
    animal = animals[0]
    trace.append({"step": 2, "claim": "animal is present in the field", "passed": True})

    # Step 3: Verify the detected object corresponds to the concept of an animal
    is_animal = animal.verify_property("object", "animal") or img.exists("animal") or img.exists("horse")
    if not is_animal:
        return {"verdict": "INVALID", "failed_step": 3, "error": "Object in the field is not an animal", "steps": trace}
    trace.append({"step": 3, "claim": "object concept relates to an animal", "passed": True})

    # Step 4: Query/verify the specific kind of animal (horse)
    is_horse = animal.verify_property("animal", "horse") or img.exists("horse")
    if not is_horse:
        animal_type = animal.query("What kind of animal is this?")
        if "horse" not in animal_type.lower():
            return {"verdict": "INVALID", "failed_step": 4, "error": f"Animal is not a horse (found: {animal_type})", "steps": trace}
    trace.append({"step": 4, "claim": "the animal in the field is a horse", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}