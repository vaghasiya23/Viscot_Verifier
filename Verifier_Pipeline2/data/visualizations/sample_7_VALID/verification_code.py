def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Locate the field in the image
    fields = img.find("field")
    if len(fields) == 0:
        fields = img.find("grass")
    if len(fields) == 0 and not img.exists("field"):
        return {
            "verdict": "INVALID",
            "failed_step": 1,
            "error": "No field found in the image",
            "steps": trace,
        }
    trace.append({"step": 1, "claim": "Field located in the image", "passed": True})

    # Step 2: Identify the animal eating from the field
    animals = img.find("animal")
    if len(animals) == 0:
        animals = img.find("horse")
    if len(animals) == 0:
        return {
            "verdict": "INVALID",
            "failed_step": 2,
            "error": "No animal found eating from the field",
            "steps": trace,
        }
    animal = animals[0]
    trace.append({"step": 2, "claim": "Animal in field located", "passed": True})

    # Step 3: Verify that the animal eating from the field is a horse
    is_horse = (
        animal.verify_property("animal", "horse")
        or img.exists("horse")
        or "horse" in animal.query("What kind of animal is this?").lower()
    )

    if not is_horse:
        return {
            "verdict": "INVALID",
            "failed_step": 3,
            "error": "The animal eating from the field is not a horse",
            "steps": trace,
        }
    trace.append({"step": 3, "claim": "The animal is a horse", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}