#!/bin/bash
# LLaDA SlowFast 批量配置测试脚本 - 任务级并行版本
# 特点：所有配置的任务混合到统一任务池，GPU动态调度，最大化利用率
# 使用方法:
#   bash batch_evaluate_configs_parallel.sh config1.yaml config2.yaml config3.yaml

echo "🔬 LLaDA SlowFast 批量配置测试 (任务级并行)"
echo "======================================"

# 检查是否提供了配置文件
if [ $# -eq 0 ]; then
    echo "❌ 错误：请提供至少一个配置文件"
    echo ""
    echo "使用方法："
    echo "  bash $0 config1.yaml config2.yaml"
    echo "  bash $0 eval/configs/*.yaml"
    exit 1
fi

# 设置环境变量
export HF_DATASETS_CACHE="/mnt/yrfs/bzh/code/LLaDA-V/train/test_data"
export HF_HUB_OFFLINE=1
export HF_ENDPOINT='https://hf-mirror.com'
export HUGGINGFACE_HUB_CACHE="/mnt/yrfs/bzh/code/LLaDA-V/train/test_data"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/mnt/yrfs/bzh/code/LLaDA-V/train/test_data"

# YAML解析函数
parse_yaml() {
    local prefix=$2
    local s='[[:space:]]*'
    local w='[a-zA-Z0-9_]*'
    local fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\):|\1|" \
         -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
         -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" $1 |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
        }
    }'
}

# 获取任务的生成参数
get_gen_kwargs() {
    local task=$1
    case $task in
        mmmu_val|mmmu_pro_standard|mmstar|ai2d|seedbench|mmbench_en_dev|mmmu_pro_vision|muirbench|videomme|mlvu_dev|mme|realworldqa)
            echo '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}'
            ;;
        chartqa)
            echo '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":16,"block_length":16,"gen_steps":16,"stopping_criteria":["\n"],"think_mode":"no_think"}'
            ;;
        docvqa_val|infovqa_val)
            echo '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":32,"block_length":32,"gen_steps":16,"think_mode":"no_think"}'
            ;;
        mathvista_testmini)
            echo '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":96,"block_length":96,"gen_steps":48,"think_mode":"think"}'
            ;;
        mathverse_testmini_vision)
            echo '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":64,"block_length":64,"gen_steps":32,"think_mode":"think"}'
            ;;
        *)
            echo '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}'
            ;;
    esac
}

# 获取所有配置文件
CONFIG_FILES=("$@")
TOTAL_CONFIGS=${#CONFIG_FILES[@]}

echo "📋 找到 $TOTAL_CONFIGS 个配置文件"

# 创建统一任务池
declare -a TASK_POOL
TASK_INDEX=0

# 解析所有配置文件，生成任务池
for config_file in "${CONFIG_FILES[@]}"; do
    if [ ! -f "$config_file" ]; then
        echo "⚠️  跳过不存在的配置文件: $config_file"
        continue
    fi

    echo "📄 解析配置: $config_file"

    # 清空之前的配置变量
    unset $(compgen -v | grep '^CONFIG_')

    # 解析当前配置
    eval $(parse_yaml "$config_file" "CONFIG_")

    # 提取配置参数
    ATTENTION_MODE="${CONFIG_attention_mode:-slowfast}"
    SLOWFAST_DOMINANT_TOKENS="${CONFIG_slowfast_dominant_tokens:-256}"
    SLOWFAST_POOLED_TOKENS="${CONFIG_slowfast_pooled_tokens:-64}"
    SLOWFAST_FULL_LAYER_INTERVAL="${CONFIG_slowfast_full_layer_interval:-4}"
    SLOWFAST_SELECTION_RATE="${CONFIG_slowfast_selection_rate:-0.1}"
    SLOWFAST_MIN_TOKENS_PER_REGION="${CONFIG_slowfast_min_tokens_per_region:-32}"
    USE_VISIONZIP="${CONFIG_use_visionzip:-false}"
    VISIONZIP_DOMINANT="${CONFIG_visionzip_dominant:-54}"
    VISIONZIP_CONTEXTUAL="${CONFIG_visionzip_contextual:-10}"
    TASK_NAMES="${CONFIG_task_names:-mmmu_val,mmstar,realworldqa}"
    BASE_MODEL_PATH="${CONFIG_base_model_path:-/mnt/yrfs/bzh/code/LLaDA-V/LLaDA-V-8B}"
    OUTPUT_BASE="${CONFIG_output_base:-exp/eval_slowfast_flexible}"
    BATCH_SIZE="${CONFIG_batch_size:-1}"
    NUM_PROCESSES="${CONFIG_num_processes:-1}"
    TOTAL_GPUS="${CONFIG_total_gpus:-8}"

    # Generate configuration identifier
    if [ "$USE_VISIONZIP" = "true" ]; then
        CONFIG_SUFFIX="visionzip_dom${VISIONZIP_DOMINANT}_ctx${VISIONZIP_CONTEXTUAL}"
    else
        CONFIG_SUFFIX="dom${SLOWFAST_DOMINANT_TOKENS}_pool${SLOWFAST_POOLED_TOKENS}_int${SLOWFAST_FULL_LAYER_INTERVAL}"
    fi
    OUTPUT_PATH="${OUTPUT_BASE}/${CONFIG_SUFFIX}"

    # 拆分任务
    IFS=',' read -ra TASKS <<< "$TASK_NAMES"

    # 为每个任务创建任务描述
    for task in "${TASKS[@]}"; do
        GEN_KWARGS=$(get_gen_kwargs "$task")

        # Build model_args
        MODEL_ARGS="pretrained=$BASE_MODEL_PATH,conv_template=llava_llada,model_name=llava_llada,attention_mode=$ATTENTION_MODE"
        # ,attn_implementation=eager"
        if [ "$ATTENTION_MODE" = "slowfast" ]; then
            MODEL_ARGS="${MODEL_ARGS},slowfast_dominant_tokens=$SLOWFAST_DOMINANT_TOKENS,slowfast_pooled_tokens=$SLOWFAST_POOLED_TOKENS,slowfast_full_layer_interval=$SLOWFAST_FULL_LAYER_INTERVAL,slowfast_selection_rate=$SLOWFAST_SELECTION_RATE,slowfast_min_tokens_per_region=$SLOWFAST_MIN_TOKENS_PER_REGION"
        fi

        # Add VisionZip parameters if enabled
        if [ "$USE_VISIONZIP" = "true" ]; then
            MODEL_ARGS="${MODEL_ARGS},use_visionzip=True,visionzip_dominant=$VISIONZIP_DOMINANT,visionzip_contextual=$VISIONZIP_CONTEXTUAL"
        fi

        # 任务描述格式：配置文件|任务名|输出路径|model_args|gen_kwargs|batch_size|num_processes|config_suffix
        TASK_POOL[$TASK_INDEX]="$config_file|$task|$OUTPUT_PATH|$MODEL_ARGS|$GEN_KWARGS|$BATCH_SIZE|$NUM_PROCESSES|$CONFIG_SUFFIX"
        TASK_INDEX=$((TASK_INDEX + 1))
    done

    echo "  ✓ 添加 ${#TASKS[@]} 个任务到任务池"
done

TOTAL_TASKS=${#TASK_POOL[@]}
echo ""
echo "======================================"
echo "📊 任务池统计："
echo "  - 配置文件数: $TOTAL_CONFIGS"
echo "  - 总任务数: $TOTAL_TASKS"
echo "  - 可用GPU数: $TOTAL_GPUS"
echo "======================================"
echo ""

if [ $TOTAL_TASKS -eq 0 ]; then
    echo "❌ 没有可执行的任务"
    exit 1
fi

# GPU 状态管理
declare -A GPU_STATUS  # 0=空闲, 1=忙碌
declare -A GPU_PIDS    # GPU上运行的任务PID
declare -A GPU_TASKS   # GPU上运行的任务描述

# 初始化GPU状态
for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
    GPU_STATUS[$gpu]=0
done

COMPLETED_TASKS=0  # 已启动的任务数
FINISHED_TASKS=0   # 已完成的任务数
BATCH_START_TIME=$(date +%s)

echo "🚀 开始任务调度..."
echo ""

# 主调度循环
while [ $FINISHED_TASKS -lt $TOTAL_TASKS ]; do
    # 检查已完成的任务并释放GPU
    for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
        if [[ ${GPU_STATUS[$gpu]} -eq 1 && -n "${GPU_PIDS[$gpu]}" ]]; then
            if ! kill -0 ${GPU_PIDS[$gpu]} 2>/dev/null; then
                echo "✅ GPU $gpu 完成任务: ${GPU_TASKS[$gpu]}"
                GPU_STATUS[$gpu]=0
                unset GPU_PIDS[$gpu]
                unset GPU_TASKS[$gpu]
                FINISHED_TASKS=$((FINISHED_TASKS + 1))

                # 计算进度
                PROGRESS=$((FINISHED_TASKS * 100 / TOTAL_TASKS))
                ELAPSED=$(($(date +%s) - BATCH_START_TIME))
                if [ $FINISHED_TASKS -gt 0 ]; then
                    AVG_TIME=$((ELAPSED / FINISHED_TASKS))
                    ETA=$(((TOTAL_TASKS - FINISHED_TASKS) * AVG_TIME))
                    echo "📈 进度: $FINISHED_TASKS/$TOTAL_TASKS ($PROGRESS%) | 预计剩余: $((ETA/60))分钟"
                fi
            fi
        fi
    done

    # 为空闲GPU分配新任务
    if [ $COMPLETED_TASKS -lt $TOTAL_TASKS ]; then
        for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
            if [[ ${GPU_STATUS[$gpu]} -eq 0 && $COMPLETED_TASKS -lt $TOTAL_TASKS ]]; then
                # 从任务池获取任务
                TASK_STRING="${TASK_POOL[$COMPLETED_TASKS]}"
                IFS='|' read -r CONFIG_FILE TASK_NAME OUTPUT_PATH MODEL_ARGS GEN_KWARGS BATCH_SIZE NUM_PROCESSES CONFIG_SUFFIX <<< "$TASK_STRING"

                # 准备输出目录
                CURRENT_OUTPUT_PATH="$OUTPUT_PATH/${TASK_NAME}"
                mkdir -p "$CURRENT_OUTPUT_PATH"

                LOG_FILE="$CURRENT_OUTPUT_PATH/${TASK_NAME}_eval.log"

                # 创建日志头
                cat > "$LOG_FILE" << EOF
=== LLaDA-V 任务级并行测试日志 ===
配置文件: $CONFIG_FILE
任务: $TASK_NAME
配置标识: $CONFIG_SUFFIX
输出路径: $CURRENT_OUTPUT_PATH
Model Args: $MODEL_ARGS
Gen Kwargs: $GEN_KWARGS
GPU ID: $gpu
任务编号: $((COMPLETED_TASKS + 1)) / $TOTAL_TASKS
开始时间: $(date)
========================================
EOF

                # 标记GPU为忙碌
                GPU_STATUS[$gpu]=1
                GPU_TASKS[$gpu]="$TASK_NAME ($CONFIG_SUFFIX)"

                echo "🔄 GPU $gpu 启动任务 $((COMPLETED_TASKS + 1))/$TOTAL_TASKS: $TASK_NAME [$CONFIG_SUFFIX]"

                # 后台执行任务
                (
                    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 accelerate launch --num_processes=$NUM_PROCESSES -m lmms_eval \
                        --model llava_onevision_llada \
                        ${GEN_KWARGS:+--gen_kwargs="$GEN_KWARGS"} \
                        --model_args "$MODEL_ARGS" \
                        --tasks "$TASK_NAME" \
                        --batch_size $BATCH_SIZE \
                        --log_samples \
                        --log_samples_suffix "${TASK_NAME}_${CONFIG_SUFFIX}" \
                        --output_path "$CURRENT_OUTPUT_PATH"

                    echo "完成时间: $(date)" >> "$LOG_FILE"
                ) >> "$LOG_FILE" 2>&1 &

                GPU_PIDS[$gpu]=$!
                COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
            fi
        done
    fi

    # 短暂等待
    if [ $FINISHED_TASKS -lt $TOTAL_TASKS ]; then
        sleep 10
    fi
done

BATCH_END_TIME=$(date +%s)
TOTAL_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 所有任务完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 统计信息："
echo "  - 总任务数: $TOTAL_TASKS"
echo "  - 配置数: $TOTAL_CONFIGS"
echo "  - 总用时: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
echo "  - 平均每任务: $((TOTAL_DURATION / TOTAL_TASKS))秒"
echo "  - 完成时间: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
