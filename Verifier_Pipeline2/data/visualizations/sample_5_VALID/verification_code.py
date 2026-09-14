def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Select/detect the "street" entity in the image
    streets = img.find("street")
    if len(streets) == 0 and not img.exists("street"):
        return {
            "verdict": "INVALID",
            "failed_step": 1,
            "error": "No street detected in the image",
            "steps": trace,
        }
    trace.append({"step": 1, "claim": "street entity exists", "passed": True})

    # Step 2: Relate the "man" in the street
    men = img.find("man")
    if len(men) == 0:
        men = img.find("person")
    if len(men) == 0:
        return {
            "verdict": "INVALID",
            "failed_step": 2,
            "error": "No man found in the street",
            "steps": trace,
        }
    man = men[0]
    trace.append({"step": 2, "claim": "man in street exists", "passed": True})

    # Step 3: Relate the action "wearing" to the man
    is_wearing = (
        man.exists("clothes")
        or man.exists("clothing")
        or man.exists("pants")
        or man.exists("shirt")
        or man.verify_property("man", "wearing clothes")
    )
    if not is_wearing:
        return {
            "verdict": "INVALID",
            "failed_step": 3,
            "error": "Man is not wearing visible clothing",
            "steps": trace,
        }
    trace.append(
        {"step": 3, "claim": "man is wearing clothing", "passed": True}
    )

    # Step 4: Query the specific item of clothing the man is wearing
    clothing_query = man.query("What item of clothing is the man wearing?")
    if not clothing_query:
        return {
            "verdict": "INVALID",
            "failed_step": 4,
            "error": "Could not query or identify clothing item",
            "steps": trace,
        }
    trace.append(
        {"step": 4, "claim": "queried clothing item worn by man", "passed": True}
    )

    return {"verdict": "VALID", "failed_step": None, "steps": trace}