def verify_reasoning(img) -> dict:
    trace = []

    # Step 1: Select the rock in the image
    rocks = img.find("rock")
    if len(rocks) == 0:
        rocks = img.find("boulder")
    if len(rocks) == 0:
        return {"verdict": "INVALID", "failed_step": 1, "error": "No rock found in the image", "steps": trace}
    rock = rocks[0]
    trace.append({"step": 1, "claim": "rock exists in the image", "passed": True})

    # Step 2: Relate the concept of a person being on the rock
    people = img.find("person")
    if len(people) == 0:
        people = img.find("child")
    
    # Filter people positioned on or above the rock
    people_on_rock = [
        p for p in people 
        if p.is_above(rock) or (p.bottom <= rock.vertical_center and p.horizontal_center >= rock.left and p.horizontal_center <= rock.right)
    ]
    
    if len(people_on_rock) == 0:
        # Check if the rock patch contains person/child directly
        if not (rock.exists("person") or rock.exists("child") or rock.exists("children")):
            return {"verdict": "INVALID", "failed_step": 2, "error": "No person found on the rock", "steps": trace}
        target_patch = rock
    else:
        target_patch = people_on_rock[0]
        
    trace.append({"step": 2, "claim": "person is located on the rock", "passed": True})

    # Step 3: Query the names/identity of the person or people on the rock
    is_children = (
        target_patch.verify_property("person", "child")
        or target_patch.verify_property("person", "children")
        or target_patch.exists("child")
        or target_patch.exists("children")
        or img.exists("children")
    )
    
    if not is_children:
        query_res = target_patch.query("Is this a child or adult?")
        if "child" not in query_res.lower() and "kid" not in query_res.lower():
            return {"verdict": "INVALID", "failed_step": 3, "error": "Could not verify identity of person on the rock", "steps": trace}

    trace.append({"step": 3, "claim": "queried and verified identity of people on the rock", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}