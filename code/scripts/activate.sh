# activate.sh：实验执行与环境配置入口。
# 对应 DESIGN.md 第 6 节。
# 参数和运行配置交由共用脚本处理。
# From the project root, first run `conda activate ./.conda` for the local environment.
PARASORT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -d "$PARASORT_ROOT/.conda/bin" ]]; then
    export PATH="$PARASORT_ROOT/.conda/bin:$PATH"
    export LD_LIBRARY_PATH="$PARASORT_ROOT/.conda/lib:${LD_LIBRARY_PATH:-}"
fi
export MPICH_CXX="${MPICH_CXX:-g++}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PARASORT_ROOT/code/results/.mpl-cache}"
