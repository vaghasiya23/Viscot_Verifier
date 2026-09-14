def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Identify the intention and check if relevant scene elements (plate/food) exist
    has_relevant_objects = img.exists("plate") or img.exists("food") or img.exists("fast food") or img.exists("hot dog")
    if not has_relevant_objects:
        return {"verdict": "INVALID", "failed_step": 1, "error": "No plate or food context found in image", "steps": trace}
    trace.append({"step": 1, "claim": "Image contains food and plate context", "passed": True})

    # Step 2: Locate the plate in the image
    plates = img.find("plate")
    if len(plates) == 0:
        return {"verdict": "INVALID", "failed_step": 2, "error": "No plate detected", "steps": trace}
    plate = plates[0]
    trace.append({"step": 2, "claim": "Plate located", "passed": True})

    # Step 3: See the fast food item placed on the plate
    food_items = img.find("fast food") + img.find("food") + img.find("hot dog")
    if len(food_items) == 0:
        food_items = plate.find("food") + plate.find("hot dog")
    if len(food_items) == 0:
        return {"verdict": "INVALID", "failed_step": 3, "error": "No fast food item found", "steps": trace}
    food_item = food_items[0]
    trace.append({"step": 3, "claim": "Fast food item seen on plate", "passed": True})

    # Step 4: Relate the fast food item to the plate (spatial overlap / inclusion)
    is_on_plate = (
        food_item.iou(plate) > 0 or
        (plate.left <= food_item.horizontal_center <= plate.right and plate.top <= food_item.vertical_center <= plate.bottom) or
        plate.exists("food") or plate.exists("hot dog")
    )
    if not is_on_plate:
        return {"verdict": "INVALID", "failed_step": 4, "error": "Fast food is not positioned on the plate", "steps": trace}
    trace.append({"step": 4, "claim": "Fast food item is related to the plate", "passed": True})

    # Step 5: Query the name of the fast food item
    queried_name = plate.query("What kind of fast food is on the plate?").lower()
    if not queried_name:
        return {"verdict": "INVALID", "failed_step": 5, "error": "Could not identify fast food item via query", "steps": trace}
    trace.append({"step": 5, "claim": "Queried fast food item name", "passed": True})

    # Step 6: Verify the fast food is a hot dog
    is_hot_dog = "hot dog" in queried_name or img.exists("hot dog") or plate.verify_property("food", "hot dog")
    if not is_hot_dog:
        return {"verdict": "INVALID", "failed_step": 6, "error": "The fast food on the plate is not a hot dog", "steps": trace}
    trace.append({"step": 6, "claim": "Fast food on the plate is a hot dog", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}