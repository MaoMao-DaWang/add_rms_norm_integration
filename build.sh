#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ASCEND_HOME_PATH:?Source the CANN set_env.sh before building}"
export CMAKE_PREFIX_PATH="${ASCEND_HOME_PATH}:${CMAKE_PREFIX_PATH:-}"
cmake -S "$project_dir" -B "$project_dir/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$(command -v python3)" \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build "$project_dir/build" --parallel "${BUILD_JOBS:-2}"
python3 "$project_dir/test_operator.py" --quick
