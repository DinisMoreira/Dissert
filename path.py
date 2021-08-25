from reeds_shepp_curves import reeds_shepp as rs
from reeds_shepp_curves import utils
import math

# a list of "vectors" (x, y, angle in degrees)
ROUTE = [(-2,4,180), (2,4,0), (2,-3,90)]#, (-5,-6,240), (-6, -7, 160), (-7,-1,80)]

full_path = []
total_length = 0

for i in range(len(ROUTE) - 1):
    path = rs.get_optimal_path(ROUTE[i], ROUTE[i+1])
    full_path += path
    total_length += rs.path_length(path)

print("Shortest path length: {}".format(round(total_length, 2)))

for e in full_path:
    print(e) 
    # e.steering (LEFT/RIGHT/STRAIGHT), e.gear (FORWARD/BACKWARD), e.param (distance)