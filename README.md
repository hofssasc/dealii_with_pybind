# Recomended use (in source dir):
cmake -S . -B build
cmake --build build
cmake --install build


ctest --test-dir build
#Runs only apps tests and outputs on failure
ctest --test-dir build -R apps --output-on-failure