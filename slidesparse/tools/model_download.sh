#!/bin/bash
# ============================================================================
# SlideSparse Model Download Script
# ============================================================================
# 用于批量下载 vLLM 基线测试所需的 W8A8 量化模型
# 支持 INT8 (quantized.w8a8) 和 FP8 (-FP8-dynamic) 两种格式
# 
# 使用方法:
#   ./model_download.sh [选项]
#   
# 选项:
#   -a, --all           下载所有模型
#   -i, --int8          仅下载 INT8 模型
#   -f, --fp8           仅下载 FP8 模型
#   -q, --qwen          仅下载 Qwen2.5 系列
#   -l, --llama         仅下载 Llama3.2 系列
#   -m, --model NAME    下载指定模型 (如: qwen2.5-7b-int8)
#   -c, --check         检查已下载模型状态
#   -h, --help          显示帮助信息
#
# 示例:
#   ./model_download.sh --all                    # 下载全部模型
#   ./model_download.sh --int8 --qwen            # 下载 Qwen INT8 模型
#   ./model_download.sh --model qwen2.5-7b-fp8   # 下载指定模型
#   ./model_download.sh --check                  # 检查下载状态
# ============================================================================


# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

# HuggingFace 组织名
HF_ORG="RedHatAI"

# ============================================================================
# 模型定义
# ============================================================================
# 格式: [简短key]="HF模型名|本地文件夹名"
# 本地文件夹命名规则: {模型系列}-{规模}-{量化类型}
# 例如: Qwen2.5-7B-FP8, Llama3.2-1B-INT8

# INT8 W8A8 模型 (quantized.w8a8)
declare -A INT8_MODELS=(
    ["qwen2.5-0.5b-int8"]="Qwen2.5-0.5B-Instruct-quantized.w8a8|Qwen2.5-0.5B-INT8"
    ["qwen2.5-1.5b-int8"]="Qwen2.5-1.5B-Instruct-quantized.w8a8|Qwen2.5-1.5B-INT8"
    ["qwen2.5-3b-int8"]="Qwen2.5-3B-Instruct-quantized.w8a8|Qwen2.5-3B-INT8"
    ["qwen2.5-7b-int8"]="Qwen2.5-7B-Instruct-quantized.w8a8|Qwen2.5-7B-INT8"
    ["qwen2.5-14b-int8"]="Qwen2.5-14B-Instruct-quantized.w8a8|Qwen2.5-14B-INT8"
    ["llama3.2-1b-int8"]="Llama-3.2-1B-Instruct-quantized.w8a8|Llama3.2-1B-INT8"
    ["llama3.2-3b-int8"]="Llama-3.2-3B-Instruct-quantized.w8a8|Llama3.2-3B-INT8"
)

# FP8 W8A8 模型 (FP8-dynamic)
declare -A FP8_MODELS=(
    ["qwen2.5-0.5b-fp8"]="Qwen2.5-0.5B-Instruct-FP8-dynamic|Qwen2.5-0.5B-FP8"
    ["qwen2.5-1.5b-fp8"]="Qwen2.5-1.5B-Instruct-FP8-dynamic|Qwen2.5-1.5B-FP8"
    ["qwen2.5-3b-fp8"]="Qwen2.5-3B-Instruct-FP8-dynamic|Qwen2.5-3B-FP8"
    ["qwen2.5-7b-fp8"]="Qwen2.5-7B-Instruct-FP8-dynamic|Qwen2.5-7B-FP8"
    ["qwen2.5-14b-fp8"]="Qwen2.5-14B-Instruct-FP8-dynamic|Qwen2.5-14B-FP8"
    ["llama3.2-1b-fp8"]="Llama-3.2-1B-Instruct-FP8-dynamic|Llama3.2-1B-FP8"
    ["llama3.2-3b-fp8"]="Llama-3.2-3B-Instruct-FP8-dynamic|Llama3.2-3B-FP8"
)

# 解析模型信息的辅助函数
get_hf_name() {
    echo "$1" | cut -d'|' -f1
}

get_local_dir_name() {
    echo "$1" | cut -d'|' -f2
}

# ============================================================================
# 辅助函数
# ============================================================================

print_header() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "SlideSparse Model Download Script"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -a, --all           下载所有模型 (INT8 + FP8)"
    echo "  -i, --int8          仅下载 INT8 (quantized.w8a8) 模型"
    echo "  -f, --fp8           仅下载 FP8 (FP8-dynamic) 模型"
    echo "  -q, --qwen          仅下载 Qwen2.5 系列"
    echo "  -l, --llama         仅下载 Llama3.2 系列"
    echo "  -m, --model NAME    下载指定模型"
    echo "  -c, --check         检查已下载模型状态"
    echo "  -s, --size          显示模型预估大小"
    echo "  -h, --help          显示此帮助信息"
    echo ""
    echo "可用模型列表:"
    echo ""
    echo "  INT8 模型 (quantized.w8a8):"
    for key in "${!INT8_MODELS[@]}"; do
        echo "    - $key"
    done | sort
    echo ""
    echo "  FP8 模型 (FP8-dynamic):"
    for key in "${!FP8_MODELS[@]}"; do
        echo "    - $key"
    done | sort
    echo ""
    echo "示例:"
    echo "  $0 --all                         # 下载全部模型"
    echo "  $0 --int8                        # 下载所有 INT8 模型"
    echo "  $0 --fp8 --qwen                  # 下载 Qwen FP8 模型"
    echo "  $0 --model qwen2.5-7b-int8       # 下载指定模型"
    echo "  $0 --check                       # 检查下载状态"
}

# 创建目录结构
setup_directories() {
    print_info "检查目录结构..."
    
    # 创建 checkpoints 根目录
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        mkdir -p "$CHECKPOINT_DIR"
        print_success "创建 checkpoints 目录: $CHECKPOINT_DIR"
    fi
}

# 下载单个模型
download_model() {
    local model_key=$1
    local model_info=$2
    local quant_type=$3  # int8 或 fp8 (仅用于显示)
    
    local hf_name=$(get_hf_name "$model_info")
    local local_dir_name=$(get_local_dir_name "$model_info")
    local hf_path="${HF_ORG}/${hf_name}"
    local local_dir="${CHECKPOINT_DIR}/${local_dir_name}"
    
    print_header "下载模型: ${local_dir_name}"
    print_info "HuggingFace 路径: ${hf_path}"
    print_info "本地保存路径: ${local_dir}"
    echo ""
    
    # 检查是否已下载
    if [ -d "$local_dir" ] && [ -f "${local_dir}/config.json" ]; then
        print_warning "模型已存在，跳过下载: ${local_dir_name}"
        echo ""
        return 0
    fi
    
    # 创建目标目录
    mkdir -p "$local_dir"
    
    # 使用 hf download 下载
    print_info "开始下载..."
    if hf download \
        "${hf_path}" \
        --local-dir "${local_dir}"; then
        print_success "下载完成: ${local_dir_name}"
    else
        print_error "下载失败: ${local_dir_name}"
        return 1
    fi
    
    echo ""
}

# 检查模型状态
check_model_status() {
    local model_info=$1
    local local_dir_name=$(get_local_dir_name "$model_info")
    local local_dir="${CHECKPOINT_DIR}/${local_dir_name}"
    
    if [ -d "$local_dir" ] && [ -f "${local_dir}/config.json" ]; then
        # 计算目录大小
        local size=$(du -sh "$local_dir" 2>/dev/null | cut -f1)
        echo -e "  ${GREEN}✓${NC} ${local_dir_name} - ${size}"
        return 0
    else
        echo -e "  ${RED}✗${NC} ${local_dir_name} - 未下载"
        return 1
    fi
}

# 检查所有模型状态
check_all_models() {
    print_header "模型下载状态检查"
    
    local downloaded=0
    local missing=0
    
    echo "INT8 模型 (quantized.w8a8):"
    echo "-----------------------------------------"
    for key in $(echo "${!INT8_MODELS[@]}" | tr ' ' '\n' | sort); do
        if check_model_status "${INT8_MODELS[$key]}"; then
            downloaded=$((downloaded + 1))
        else
            missing=$((missing + 1))
        fi
    done
    
    echo ""
    echo "FP8 模型 (FP8-dynamic):"
    echo "-----------------------------------------"
    for key in $(echo "${!FP8_MODELS[@]}" | tr ' ' '\n' | sort); do
        if check_model_status "${FP8_MODELS[$key]}"; then
            downloaded=$((downloaded + 1))
        else
            missing=$((missing + 1))
        fi
    done
    
    echo ""
    echo "-----------------------------------------"
    echo -e "总计: ${GREEN}${downloaded} 已下载${NC}, ${RED}${missing} 未下载${NC}"
    echo ""
    
    # 显示 checkpoints 总大小
    if [ -d "$CHECKPOINT_DIR" ]; then
        local total_size=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
        print_info "Checkpoints 目录总大小: ${total_size}"
    fi
}

# 显示模型预估大小
show_model_sizes() {
    print_header "模型预估大小"
    echo "注意: 以下为大致估算值，实际大小可能有所不同"
    echo ""
    echo "模型规模参考:"
    echo "  - 0.5B 模型: ~1-2 GB"
    echo "  - 1B 模型:   ~2-3 GB"
    echo "  - 1.5B 模型: ~3-4 GB"
    echo "  - 3B 模型:   ~6-8 GB"
    echo "  - 7B 模型:   ~14-16 GB"
    echo "  - 14B 模型:  ~28-32 GB"
    echo ""
    echo "预估总大小 (全部模型):"
    echo "  - INT8 全部: ~65-80 GB"
    echo "  - FP8 全部:  ~65-80 GB"
    echo "  - 总计:      ~130-160 GB"
}

# 下载 INT8 模型
download_int8_models() {
    local filter=$1  # qwen, llama, or empty for all
    
    for key in $(echo "${!INT8_MODELS[@]}" | tr ' ' '\n' | sort); do
        local model_info="${INT8_MODELS[$key]}"
        
        # 过滤
        if [ -n "$filter" ]; then
            case $filter in
                qwen)
                    [[ ! $key == qwen* ]] && continue
                    ;;
                llama)
                    [[ ! $key == llama* ]] && continue
                    ;;
            esac
        fi
        
        download_model "$key" "$model_info" "int8"
    done
}

# 下载 FP8 模型
download_fp8_models() {
    local filter=$1  # qwen, llama, or empty for all
    
    for key in $(echo "${!FP8_MODELS[@]}" | tr ' ' '\n' | sort); do
        local model_info="${FP8_MODELS[$key]}"
        
        # 过滤
        if [ -n "$filter" ]; then
            case $filter in
                qwen)
                    [[ ! $key == qwen* ]] && continue
                    ;;
                llama)
                    [[ ! $key == llama* ]] && continue
                    ;;
            esac
        fi
        
        download_model "$key" "$model_info" "fp8"
    done
}

# 下载指定模型
download_specific_model() {
    local model_key=$1
    
    # 检查是否在 INT8 模型列表中
    if [ -n "${INT8_MODELS[$model_key]}" ]; then
        download_model "$model_key" "${INT8_MODELS[$model_key]}" "int8"
        return 0
    fi
    
    # 检查是否在 FP8 模型列表中
    if [ -n "${FP8_MODELS[$model_key]}" ]; then
        download_model "$model_key" "${FP8_MODELS[$model_key]}" "fp8"
        return 0
    fi
    
    print_error "未找到模型: $model_key"
    echo "使用 --help 查看可用模型列表"
    return 1
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    # 检查 hf CLI 是否安装（新版命令）
    if ! command -v hf &> /dev/null; then
        # 回退检查旧版命令
        if ! command -v huggingface-cli &> /dev/null; then
            print_error "huggingface CLI 未安装"
            print_info "请运行: pip install -U huggingface_hub"
            exit 1
        fi
    fi
    
    # 解析参数
    local download_int8=false
    local download_fp8=false
    local filter_qwen=false
    local filter_llama=false
    local specific_model=""
    local check_only=false
    local show_sizes=false
    
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--all)
                download_int8=true
                download_fp8=true
                shift
                ;;
            -i|--int8)
                download_int8=true
                shift
                ;;
            -f|--fp8)
                download_fp8=true
                shift
                ;;
            -q|--qwen)
                filter_qwen=true
                shift
                ;;
            -l|--llama)
                filter_llama=true
                shift
                ;;
            -m|--model)
                specific_model="$2"
                shift 2
                ;;
            -c|--check)
                check_only=true
                shift
                ;;
            -s|--size)
                show_sizes=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知选项: $1"
                echo "使用 --help 查看帮助"
                exit 1
                ;;
        esac
    done
    
    # 显示大小信息
    if [ "$show_sizes" = true ]; then
        show_model_sizes
        exit 0
    fi
    
    # 检查模式
    if [ "$check_only" = true ]; then
        check_all_models
        exit 0
    fi
    
    # 设置目录
    setup_directories
    
    # 下载指定模型
    if [ -n "$specific_model" ]; then
        download_specific_model "$specific_model"
        exit $?
    fi
    
    # 确定过滤器
    local filter=""
    if [ "$filter_qwen" = true ] && [ "$filter_llama" = false ]; then
        filter="qwen"
    elif [ "$filter_llama" = true ] && [ "$filter_qwen" = false ]; then
        filter="llama"
    fi
    
    # 执行下载
    if [ "$download_int8" = true ]; then
        print_header "开始下载 INT8 (quantized.w8a8) 模型"
        download_int8_models "$filter"
    fi
    
    if [ "$download_fp8" = true ]; then
        print_header "开始下载 FP8 (FP8-dynamic) 模型"
        download_fp8_models "$filter"
    fi
    
    # 显示最终状态
    check_all_models
    
    print_success "模型下载任务完成!"
}

# 运行主程序
main "$@"
