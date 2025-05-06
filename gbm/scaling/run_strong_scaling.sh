#!/bin/bash

LOG_FILE="test_strong.log"

echo "Running strong scaling study..."
echo "Logs will be written to: $LOG_FILE"

for i in {0..4};
do
    echo "Running with $((2**i)) processes..."
    mpirun -np $((2**i)) python3 strong_scaler.py --log $LOG_FILE --adjoint --work 1000000
done
