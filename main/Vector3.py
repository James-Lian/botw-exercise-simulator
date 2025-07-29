import math
import numpy as np
class Vector3: # can be used to represent 3D velocities, positions, etc.
        def __init__(self, x: float, y: float, z: float):
            self.x = x
            self.y = y
            if z is not None:
                self.z = z
        
        def distance_to(self, point): # takes Vector3 or tuple as input
            if isinstance(point, type(self)):
                x = point.x
                y = point.y
                z = point.z
            elif isinstance(point, (list, tuple)):
                x = point[0]
                y = point[1]
                z = point[2]
            else:
                raise TypeError("Vector3.distance_to() expected Vector3, list, or tuple as parameter")

            return (abs(x - self.x) ** 2 + abs(y - self.y)**2 + abs(z - self.z) ** 2) ** 0.5

        def angle_between_points(self, point1, point2):
            # current Vector3 point must be in the middle of two other points, e.g. if current point is B, then measures angle ABC
            
            ## Constructing an imaginary 2D triangle by computing distances between all 3 points
            # side lengths
            a = point1.distance_to(point2) # opposite self (A)
            b = self.distance_to(point2) # opposite point1 (B)
            c = self.distance_to(point1) # opposite point2 (C)

            return math.acos((b**2 + c**2 - a**2) / (2 * b * c)) # return angle self (A)

        def angle_to_another(self, a):
            if isinstance(a, type(self)):
                a = np.array([a.x, a.y, a.z])
            elif isinstance(a, (list, tuple)):
                a = np.array(a)
            b = np.array([self.x, self.y, self.z])
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            cos_theta = dot_product / (norm_a * norm_b)
            angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid domain errors
            return float(angle_rad)
    
        def tuple(self):
            return (self.x, self.y, self.z) 
