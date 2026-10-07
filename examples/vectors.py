import numpy as np

from fibomat.linalg import Vector, angle_between

# Vector
null_vector = Vector()
vector_1 = Vector(x=1, y=2)
vector_2 = Vector(1, 2)
vector_3 = Vector(np.array([1, 2]))
vector_4 = Vector([1, 2])
vector_5 = Vector((1, 2))
vector_6 = Vector(r=1, phi=np.pi)
vector_7 = Vector(Vector(1, 2))

u = Vector(1, 2)
v = Vector(3, 4)

print(u + v)  # result in Vector(4, 6)
print(u - v)  # result in Vector(-2, -2)
print(4 * u)  # result in Vector(4, 8)
print(u / 2)  # result in Vector(0.5, 1)

print(u + (1, 2))  # results also in Vector(4, 6)
print(u + [1, 2])  # results also in Vector(4, 6)
print(u + np.array((1, 2)))  # results also in Vector(4, 6)

print(u.dot(v))  # prints the dot product between u and v, in this case 9

print(u.x, u.y)  # prints "1, 2"
print(u[0], u[1])  # prints "1, 2", too. u[0] = u.x, u[1] = u.y

w = Vector(r=1, phi=np.pi)
print(u.r, u.phi)  # prints 1, 3.14159

print(v.length)  # prints out the norm (length) of the vector
print(v.angle_about_x_axis)  # prints the angle of the vector and the positive x-axis. the result will be in [0, 2pi]

print(u.close_to(v))  # returns True, if u is nearly v and otherwise False (here: False).

print(angle_between(u, v))  # prints the angle between u and v

u_rot = u.rotated(np.pi/2)  # rotated the vector counterclockwise by np.pi/2 around the origin.
u_mir = u.mirrored([1, 0])  # mirrors the vector at the positive x-axis
u_norm = u.normalized()  # returns a vector pointing in the same direction as `u` but with length = 1

np_array = np.asarray(u)

