def is_inside_kidney(stone_box, kidney_boxes):
    sx_min, sy_min, sx_max, sy_max = stone_box
    for kx_min, ky_min, kx_max, ky_max in kidney_boxes:
        if sx_min >= kx_min and sx_max <= kx_max and sy_min >= ky_min and sy_max <= ky_max:
            return True
    return False

# Filter stones
filtered_stones = [box for box in stone_detections if is_inside_kidney(box, kidney_boxes)]
