def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Identify the cups in the image
    cups = img.find("cup")
    if len(cups) == 0:
        cups = img.find("cups")
    if len(cups) == 0:
        return {
            "verdict": "INVALID",
            "failed_step": 1,
            "error": "No cups detected in the image",
            "steps": trace,
        }
    trace.append({"step": 1, "claim": "cups exist in the image", "passed": True})

    # Step 2: Observe relationship between the cups and furniture they are hanging on
    furnitures = img.find("furniture")
    if len(furnitures) == 0:
        furnitures = (
            img.find("cabinet")
            + img.find("cupboard")
            + img.find("rack")
            + img.find("shelf")
        )

    # Filter for furniture close to or containing the cups
    related_furniture = []
    for f in furnitures:
        for c in cups:
            if f.distance(c) < 50 or f.iou(c) > 0:
                related_furniture.append(f)
                break

    if len(related_furniture) == 0:
        # Fallback to general cabinet search
        cabinets = img.find("cabinet")
        if len(cabinets) > 0:
            related_furniture = cabinets

    if len(related_furniture) == 0:
        return {
            "verdict": "INVALID",
            "failed_step": 2,
            "error": "No furniture found in proximity to cups",
            "steps": trace,
        }
    target_furniture = related_furniture[0]
    trace.append(
        {
            "step": 2,
            "claim": "cups are associated with a piece of furniture",
            "passed": True,
        }
    )

    # Step 3: Extract/verify the type of furniture is a cabinet
    is_cabinet = (
        target_furniture.verify_property("furniture", "cabinet")
        or img.exists("cabinet")
        or target_furniture.verify_property("cabinet", "cabinet")
    )

    if not is_cabinet:
        furniture_type = target_furniture.query(
            "What kind of furniture is this?"
        ).lower()
        if "cabinet" in furniture_type or "cupboard" in furniture_type:
            is_cabinet = True

    if not is_cabinet:
        return {
            "verdict": "INVALID",
            "failed_step": 3,
            "error": "The furniture is not a cabinet",
            "steps": trace,
        }

    trace.append(
        {
            "step": 3,
            "claim": "the furniture the cups are hanging on is a cabinet",
            "passed": True,
        }
    )

    return {"verdict": "VALID", "failed_step": None, "steps": trace}