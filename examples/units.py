from fibomat.linalg import DimVector, Vector
from fibomat.units import scale_factor, scale_to, unit


length_unit = unit('µm')
dose_unit = unit('ions / nm**2')
print(length_unit, dose_unit)

length = 1 * unit('nm')
dose = 10 * unit('ions / nm**2')
another_length = 10 * length_unit
print(length, dose, another_length)

# three version to create a dimensioned vector
dim_vector = Vector(3, 4) * unit('µm')
dim_vector2 = Vector(3, 4) * unit('µm')
dim_vector3 = DimVector(3 * unit('µm'), 4 * unit('µm'))
print(dim_vector, dim_vector2, dim_vector3)

length_in_um = scale_to(unit('µm'), length)  # NOTE: length_in_um is a float now and NOT a dimensioned value anymore
nm_to_um = scale_factor(unit('µm'), unit('nm'))  # scale factor (float) to scale from nm to µm
print(length_in_um, nm_to_um)

