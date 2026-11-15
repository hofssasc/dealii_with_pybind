# Basic checks of the dealii.Vector bindings. The printed output is compared
# against vector.output by the testsuite (see pybind_pickup_tests).

import dealii_core_module as dealii

v = dealii.Vector(4)
print(f"size: {v.size()}, len: {len(v)}")
print(f"all_zero: {v.all_zero()}")

for i in range(len(v)):
    v[i] = i + 1.0
print(f"entries: {[v[i] for i in range(len(v))]}")

print(f"l1_norm: {v.l1_norm()}")
print(f"l2_norm: {v.l2_norm()}")
print(f"linfty_norm: {v.linfty_norm()}")
print(f"mean_value: {v.mean_value()}")

w = dealii.Vector(v)
w *= 2.0
print(f"v * w: {v * w}")

w.axpy(-2.0, v)
print(f"after axpy, all_zero: {w.all_zero()}")
