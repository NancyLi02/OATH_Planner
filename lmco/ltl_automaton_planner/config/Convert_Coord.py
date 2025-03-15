def convert_coordinate(old_x, old_y):
    """
    Convert a point (old_x, old_y) in the original 0~8 grid
    to a new coordinate system centered at (0,0) with a range of -25 to 25.
    """
    # Define the center of the original grid
    center_x, center_y = 4, 4
    
    # Define the scaling factor that maps -4~4 to -25~25
    scale = 25 / 4  # 6.25
    
    # Calculate the new coordinates
    new_x = (old_x - center_x) * scale
    new_y = (old_y - center_y) * scale
    
    return new_x, new_y

if __name__ == "__main__":
    # Test the function with some sample points
    test_points = [(1.111, 3.087), (4, 4), (8, 8), (0, 8), (8, 0)]
    for p in test_points:
        converted = convert_coordinate(p[0], p[1])
        print(f"Original: {p} -> Converted: {converted}")
