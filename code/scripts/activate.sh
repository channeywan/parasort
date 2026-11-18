# Source after `conda activate /home/wcl/parasort/.conda` if using the project-local environment.
PARASORT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -d "$PARASORT_ROOT/.conda/bin" ]]; then
    export PATH="$PARASORT_ROOT/.conda/bin:$PATH"
    export LD_LIBRARY_PATH="$PARASORT_ROOT/.conda/lib:${LD_LIBRARY_PATH:-}"
fi
export MPICH_CXX="${MPICH_CXX:-g++}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PARASORT_ROOT/code/results/.mpl-cache}"
