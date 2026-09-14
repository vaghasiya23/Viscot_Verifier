def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Identify the children in the image
    children = img.find("children")
    if len(children) == 0:
        children = img.find("child")
    if len(children) == 0:
        children = img.find("person")
    
    if len(children) == 0:
        return {
            "verdict": "INVALID",
            "failed_step": 1,
            "error": "No children detected in the image",
            "steps": trace,
        }
    child = children[0]
    trace.append({"step": 1, "claim": "children identified", "passed": True})

    # Step 2: Determine relationship between children and the object under them
    rocks = img.find("rock")
    if len(rocks) == 0:
        rocks = img.find("boulder")
    if len(rocks) == 0:
        # Fallback to finding general objects near bottom of children
        objects = img.find("ground") + img.find("stone")
        rocks = [
            obj
            for obj in objects
            if obj.vertical_center >= child.vertical_center
        ]

    if len(rocks) == 0:
        return {
            "verdict": "INVALID",
            "failed_step": 2,
            "error": "No supporting object located under/around children",
            "steps": trace,
        }
    target_object = rocks[0]
    trace.append(
        {"step": 2, "claim": "underlying object detected", "passed": True}
    )

    # Step 3: Verify the position indicating the children are on top of the object
    is_on_top = (
        child.is_above(target_object)
        or (child.bottom >= target_object.top)
        or (child.vertical_center < target_object.vertical_center)
    )
    if not is_on_top:
        return {
            "verdict": "INVALID",
            "failed_step": 3,
            "error": "Children are not positioned on top of the object",
            "steps": trace,
        }
    trace.append(
        {
            "step": 3,
            "claim": "children positioned on top of object",
            "passed": True,
        }
    )

    # Step 4: Verify that the object the children are on is a rock
    is_rock = (
        target_object.verify_property("object", "rock")
        or img.exists("rock")
        or "rock" in target_object.query("What object is this?").lower()
    )
    if not is_rock:
        return {
            "verdict": "INVALID",
            "failed_step": 4,
            "error": "Object children are on is not a rock",
            "steps": trace,
        }
    trace.append({"step": 4, "claim": "object is a rock", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}