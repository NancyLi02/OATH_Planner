#!/usr/bin/env python
"""
Coordinate Transformation Module for TurtleBot3 Navigation

This module transforms coordinates from the pygame simulation map to real-world
coordinates (in meters) for sending navigation goals to the TurtleBot3.

Usage:
    from ltl_automaton_planner.coordinate_transform import CoordinateTransformer
    
    # Create transformer with default settings (15x12 pygame map, 15ft x 12ft real map)
    transformer = CoordinateTransformer()
    
    # Or customize the parameters
    transformer = CoordinateTransformer(
        pygame_map_size=(15, 12),       # pygame map dimensions
        real_map_size_ft=(15.0, 12.0),  # real map size in feet
        pygame_origin=(0, 0),           # pygame coordinate origin
        real_origin=(0.0, 0.0)          # real world origin offset in meters
    )
    
    # Transform coordinates
    pygame_coord = (5.0, 10.0)
    real_coord = transformer.pygame_to_real(pygame_coord)
"""

# Conversion constant: 1 foot = 0.3048 meters
FEET_TO_METERS = 0.3048


class CoordinateTransformer:
    """
    A class to transform coordinates between pygame simulation map and real-world coordinates.
    
    Attributes:
        pygame_map_size (tuple): Size of the pygame map (width, height) in units
        real_map_size_ft (tuple): Size of the real map (width, height) in feet
        real_map_size_m (tuple): Size of the real map (width, height) in meters
        pygame_origin (tuple): Origin point of the pygame coordinate system
        real_origin (tuple): Origin offset of the real world coordinate system in meters
        scale_x (float): Scaling factor for x-axis (meters per pygame unit)
        scale_y (float): Scaling factor for y-axis (meters per pygame unit)
    """
    
    def __init__(self, 
                 pygame_map_size=(15, 12), 
                 real_map_size_ft=(15.0, 12.0),
                 pygame_origin=(0, 0),
                 real_origin=(0.0, 0.0)):
        """
        Initialize the coordinate transformer.
        
        Args:
            pygame_map_size (tuple): Size of the pygame map (width, height) in units.
                                    Default: (15, 12)
            real_map_size_ft (tuple): Size of the real map (width, height) in feet.
                                      Default: (9.0, 9.0)
            pygame_origin (tuple): Origin point (x, y) of the pygame coordinate system.
                                   Default: (0, 0)
            real_origin (tuple): Origin offset (x, y) of the real world coordinate system in meters.
                                 This allows shifting the entire real-world coordinate frame.
                                 Default: (0.0, 0.0)
        """
        self.pygame_map_size = pygame_map_size
        self.real_map_size_ft = real_map_size_ft
        self.pygame_origin = pygame_origin
        self.real_origin = real_origin
        
        # Convert feet to meters
        self.real_map_size_m = (
            real_map_size_ft[0] * FEET_TO_METERS,
            real_map_size_ft[1] * FEET_TO_METERS
        )
        
        # Calculate scaling factors (meters per pygame unit)
        self.scale_x = self.real_map_size_m[0] / pygame_map_size[0]
        self.scale_y = self.real_map_size_m[1] / pygame_map_size[1]
        
    def pygame_to_real(self, pygame_coord):
        """
        Transform pygame coordinates to real-world coordinates in meters.
        
        Args:
            pygame_coord (tuple): Coordinates (x, y) in pygame units
            
        Returns:
            tuple: Coordinates (x, y) in real-world meters
        """
        pygame_x, pygame_y = pygame_coord
        
        # Apply origin offset and scaling
        real_x = (pygame_x - self.pygame_origin[0]) * self.scale_x + self.real_origin[0]
        real_y = (pygame_y - self.pygame_origin[1]) * self.scale_y + self.real_origin[1]
        
        return (real_x, real_y)
    
    def real_to_pygame(self, real_coord):
        """
        Transform real-world coordinates (in meters) to pygame coordinates.
        
        Args:
            real_coord (tuple): Coordinates (x, y) in real-world meters
            
        Returns:
            tuple: Coordinates (x, y) in pygame units
        """
        real_x, real_y = real_coord
        
        # Reverse the transformation
        pygame_x = (real_x - self.real_origin[0]) / self.scale_x + self.pygame_origin[0]
        pygame_y = (real_y - self.real_origin[1]) / self.scale_y + self.pygame_origin[1]
        
        return (pygame_x, pygame_y)
    
    def get_scale_factors(self):
        """
        Get the scaling factors used for coordinate transformation.
        
        Returns:
            tuple: (scale_x, scale_y) in meters per pygame unit
        """
        return (self.scale_x, self.scale_y)
    
    def get_real_map_size_meters(self):
        """
        Get the real map size in meters.
        
        Returns:
            tuple: (width, height) in meters
        """
        return self.real_map_size_m
    
    def update_real_map_size(self, new_size_ft):
        """
        Update the real map size (in feet) and recalculate scaling factors.
        
        Args:
            new_size_ft (tuple): New real map size (width, height) in feet
        """
        self.real_map_size_ft = new_size_ft
        self.real_map_size_m = (
            new_size_ft[0] * FEET_TO_METERS,
            new_size_ft[1] * FEET_TO_METERS
        )
        self.scale_x = self.real_map_size_m[0] / self.pygame_map_size[0]
        self.scale_y = self.real_map_size_m[1] / self.pygame_map_size[1]
        
    def update_pygame_map_size(self, new_size):
        """
        Update the pygame map size and recalculate scaling factors.
        
        Args:
            new_size (tuple): New pygame map size (width, height) in units
        """
        self.pygame_map_size = new_size
        self.scale_x = self.real_map_size_m[0] / new_size[0]
        self.scale_y = self.real_map_size_m[1] / new_size[1]
    
    def __repr__(self):
        return (f"CoordinateTransformer(\n"
                f"  pygame_map_size={self.pygame_map_size},\n"
                f"  real_map_size_ft={self.real_map_size_ft},\n"
                f"  real_map_size_m={self.real_map_size_m},\n"
                f"  pygame_origin={self.pygame_origin},\n"
                f"  real_origin={self.real_origin},\n"
                f"  scale=(x: {self.scale_x:.6f} m/unit, y: {self.scale_y:.6f} m/unit)\n"
                f")")


# Default transformer instance with standard settings
# pygame map: 15x12 units
# real map: 15ft x 12ft = 4.572m x 3.6576m
# Scale: 0.3048 m/unit (1 ft per pygame unit)
default_transformer = CoordinateTransformer()


def create_transformer_from_params(pygame_width=15, pygame_height=12,
                                    real_width_ft=15.0, real_height_ft=12.0,
                                    pygame_origin_x=0, pygame_origin_y=0,
                                    real_origin_x=0.0, real_origin_y=0.0):
    """
    Factory function to create a CoordinateTransformer from individual parameters.
    This is useful when loading parameters from ROS2 parameter server.
    
    Args:
        pygame_width (int/float): Width of pygame map in units
        pygame_height (int/float): Height of pygame map in units
        real_width_ft (float): Width of real map in feet
        real_height_ft (float): Height of real map in feet
        pygame_origin_x (int/float): X coordinate of pygame origin
        pygame_origin_y (int/float): Y coordinate of pygame origin
        real_origin_x (float): X offset of real world origin in meters
        real_origin_y (float): Y offset of real world origin in meters
        
    Returns:
        CoordinateTransformer: Configured transformer instance
    """
    return CoordinateTransformer(
        pygame_map_size=(pygame_width, pygame_height),
        real_map_size_ft=(real_width_ft, real_height_ft),
        pygame_origin=(pygame_origin_x, pygame_origin_y),
        real_origin=(real_origin_x, real_origin_y)
    )


if __name__ == "__main__":
    # Test the coordinate transformer
    print("Testing CoordinateTransformer...")
    print("=" * 50)
    
    # Create default transformer
    transformer = CoordinateTransformer()
    print(transformer)
    print()
    
    # Test some coordinate transformations
    test_coords = [(0, 0), (9, 9), (18, 18), (5, 10), (0, 18), (18, 0)]
    
    print("Pygame to Real-world transformations:")
    print("-" * 50)
    for pygame_coord in test_coords:
        real_coord = transformer.pygame_to_real(pygame_coord)
        print(f"  Pygame {pygame_coord} -> Real {real_coord[0]:.4f}m, {real_coord[1]:.4f}m")
    
    print()
    print("Reverse transformation test:")
    print("-" * 50)
    for pygame_coord in test_coords:
        real_coord = transformer.pygame_to_real(pygame_coord)
        back_to_pygame = transformer.real_to_pygame(real_coord)
        print(f"  Original: {pygame_coord}, Transformed back: ({back_to_pygame[0]:.4f}, {back_to_pygame[1]:.4f})")
    
    print()
    print("Custom transformer example (10ft x 10ft real map):")
    print("-" * 50)
    custom_transformer = CoordinateTransformer(
        pygame_map_size=(18, 18),
        real_map_size_ft=(10.0, 10.0),
        real_origin=(1.0, 1.0)  # Offset the origin by 1m in both directions
    )
    print(custom_transformer)
    pygame_coord = (9, 9)  # Center of pygame map
    real_coord = custom_transformer.pygame_to_real(pygame_coord)
    print(f"  Center of pygame map {pygame_coord} -> Real {real_coord[0]:.4f}m, {real_coord[1]:.4f}m")
