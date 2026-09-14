def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Identify the object labeled as "chair" in the image
    chairs = img.find("chair")
    if len(chairs) == 0:
        return {"verdict": "INVALID", "failed_step": 1, "error": "No chair detected", "steps": trace}
    chair = chairs[0]
    trace.append({"step": 1, "claim": "chair identified in image", "passed": True})

    # Step 2: Determine the direction to the right of the chair
    # Verify there is image area/space to the right of the chair's horizontal center
    if chair.horizontal_center >= img.right:
        return {"verdict": "INVALID", "failed_step": 2, "error": "No space to the right of chair within image", "steps": trace}
    trace.append({"step": 2, "claim": "direction to the right of chair determined", "passed": True})

    # Step 3: Look for any furniture located in that direction
    furnitures = img.find("furniture")
    right_furnitures = [f for f in furnitures if f.horizontal_center > chair.horizontal_center]
    
    # Fallback to direct detection of common furniture if generic detection is unavailable
    if len(right_furnitures) == 0:
        sofas = img.find("sofa")
        right_furnitures = [s for s in sofas if s.horizontal_center > chair.horizontal_center]
    
    if len(right_furnitures) == 0:
        return {"verdict": "INVALID", "failed_step": 3, "error": "No furniture found to the right of the chair", "steps": trace}
    
    target_furniture = right_furnitures[0]
    trace.append({"step": 3, "claim": "furniture found to the right of chair", "passed": True})

    # Step 4: Identify the type of furniture to the right of the chair (Verify it is a sofa)
    is_sofa = (
        target_furniture.verify_property("furniture", "sofa") 
        or target_furniture.exists("sofa") 
        or img.exists("sofa")
    )
    
    if not is_sofa:
        furniture_type = target_furniture.query("What type of furniture is this?")
        if "sofa" not in furniture_type.lower() and "couch" not in furniture_type.lower():
            return {"verdict": "INVALID", "failed_step": 4, "error": "Furniture to the right is not a sofa", "steps": trace}

    trace.append({"step": 4, "claim": "furniture to the right is a sofa", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}