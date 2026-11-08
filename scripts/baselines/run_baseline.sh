#!/bin/bash
# Train and evaluate one nerfstudio baseline on one plot, with the settings of the paper (NVS table, Table 2).
# Run inside the baselines Docker image (docker/baselines), after scripts/baselines/prepare_data.py for instant-ngp
# and for splatfacto with --sparse_pc.
#
#   bash scripts/baselines/run_baseline.sh <method> <plot_dir> <output_dir> [--prep_dir DIR] [--wandb] [-- <ns-train args>]
#
#   method      nerfacto | splatfacto | instant-ngp | fruit_nerf | fruit_nerf_big
#   plot_dir    plot folder with images/ and transforms.json (+ semantics/ for FruitNeRF), or prepared with --prep_dir
#   output_dir  runs go to <output_dir>/<plot>/<method>/<timestamp>/, metrics in test_results.json there
#   --prep_dir  where prepare_data.py wrote its files (its --output_dir/<plot>; default: the plot folder);
#               a transforms.json there is used instead of the plot's
#   --sparse_pc splatfacto: start from the COLMAP points (sparse_pc.ply from prepare_data.py --sparse_pc), runs in
#               <plot>/splatfacto-sparse_pc/splatfacto/; default: random points, as in the paper for plots 462-467
#   --wandb     log to Weights & Biases (project WheatGS-nerfstudio) instead of tensorboard
#   after --     extra arguments for ns-train, e.g. --max-num-iterations 1000
set -euo pipefail

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ $# -ge 3 ] || usage
method=$1 plot_dir=${2%/} output_dir=${3%/}; shift 3
prep_dir=$plot_dir vis=tensorboard sparse_pc=0 extra=()
while [ $# -gt 0 ]; do
    case $1 in
        --prep_dir) prep_dir=${2%/}; shift 2 ;;
        --sparse_pc) sparse_pc=1; shift ;;
        --wandb) vis=wandb; shift ;;
        --) shift; extra=("$@"); break ;;
        *) usage ;;
    esac
done

plot=$(basename "$plot_dir")
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_dir=$output_dir/$plot/$method/$timestamp
out=(--output-dir "$output_dir" --experiment-name "$plot")  # nerfstudio writes <output-dir>/<experiment>/<method>/<timestamp>
tf=$prep_dir/transforms.json  # from prepare_data.py (no transforms.json in the plot, or other semantics)
[ -f "$tf" ] || tf=$plot_dir/transforms.json
# Apptainer: triton (torch.compile, used by splatfacto only for its view matrix) finds libcuda.so only via the ldconfig
# cache, which --nv doesn't update; without compiling, the results are the same
[ -d /.singularity.d ] && export TORCHDYNAMO_DISABLE=1
# nerfacc creates its build folder (default ~/.cache/torch_extensions) on import; the home can be read-only in
# Apptainer on clusters (e.g. Euler), so default to /tmp
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions_$(id -u)}
ns=()  # nerfstudio 1.1.5 of the image; FruitNeRF has its own env (nerfstudio 0.3.2)

case $method in
    nerfacto)
        args=(--data "$tf" --pipeline.model.camera-optimizer.mode off
              nerfstudio-data --train-split-fraction 1.0) ;;
    splatfacto)
        if [ $sparse_pc = 1 ]; then  # own folder, so it doesn't mix with the random-init runs
            grep -q '"ply_file_path"' "$tf" || { echo "$tf has no ply_file_path: run prepare_data.py --sparse_pc"; exit 1; }
            init=()
            out=(--output-dir "$output_dir/$plot" --experiment-name splatfacto-sparse_pc)
            run_dir=$output_dir/$plot/splatfacto-sparse_pc/splatfacto/$timestamp
        else
            init=(--pipeline.model.random-init True)
        fi
        args=("${init[@]}" --data "$tf" nerfstudio-data) ;;
    instant-ngp)
        args=(--data "$prep_dir" blender-data) ;;  # blender-data ignores cx/cy (see README)
    fruit_nerf|fruit_nerf_big)
        ns=(conda run --no-capture-output -n fruitnerf)
        args=(--data "$tf" pipeline:fruit-pipeline-config
              --pipeline.datamanager.camera-optimizer.mode off) ;;
    *) echo "Unknown method: $method"; usage ;;
esac

echo "== $method on $plot -> $run_dir"
start=$(date +%s)
"${ns[@]}" ns-train "$method" "${out[@]}" --timestamp "$timestamp" --vis "$vis" --project-name WheatGS-nerfstudio \
    "${extra[@]}" "${args[@]}"
echo $(( $(date +%s) - start )) > "$run_dir/train_seconds.txt"
"${ns[@]}" ns-eval --load-config "$run_dir/config.yml" --output-path "$run_dir/test_results.json"
echo "== done: $run_dir/test_results.json"
