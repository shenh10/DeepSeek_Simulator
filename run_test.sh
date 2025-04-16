#!/bin/bash
set -e

OUTPUT_FOLDER=${OUTPUT_FOLDER:-"results"}
PREFIX=${PREFIX:-"MI300X"}

function parse_mla_result_to_csv {
    # Create output CSV file with header
    local output_file=$1
    echo "b,s_q,mean_sk,h_q,h_kv,d,dv,causal,varlen,latency,tflops,bandwidth" > "$output_file"  

    # Process the input data  
    while read line; do  
        b=$(echo "$line" | grep -o 'b=[0-9]*' | cut -d= -f2)  
        s_q=$(echo "$line" | grep -o 's_q=[0-9]*' | cut -d= -f2)  
        mean_sk=$(echo "$line" | grep -o 'mean_sk=[0-9]*' | cut -d= -f2)  
        h_q=$(echo "$line" | grep -o 'h_q=[0-9]*' | cut -d= -f2)  
        h_kv=$(echo "$line" | grep -o 'h_kv=[0-9]*' | cut -d= -f2)  
        d=$(echo "$line" | grep -o 'd=[0-9]*' | cut -d= -f2)  
        dv=$(echo "$line" | grep -o 'dv=[0-9]*' | cut -d= -f2)  
        causal=$(echo "$line" | grep -o 'causal=[A-Za-z]*' | cut -d= -f2)  
        varlen=$(echo "$line" | grep -o 'varlen=[A-Za-z]*' | cut -d= -f2)  
        latency=$(echo "$line" | grep -o '[0-9.]\+ ms' | cut -d' ' -f1)  
        tflops=$(echo "$line" | grep -o '[0-9]\+ TFLOPS' | cut -d' ' -f1)  
        bandwidth=$(echo "$line" | grep -o '[0-9]\+ GB/s' | cut -d' ' -f1)  

        echo "$b,$s_q,$mean_sk,$h_q,$h_kv,$d,$dv,$causal,$varlen,$latency,$tflops,$bandwidth" >> "$output_file"  
    done
}


if [ ! -d "$OUTPUT_FOLDER" ]; then  
    mkdir -p "$OUTPUT_FOLDER"  
fi  

mla_out=$OUTPUT_FOLDER/${PREFIX}_mla.csv
dense_gemm_out=$OUTPUT_FOLDER/${PREFIX}_dense_gemm.csv
group_gemm_out=$OUTPUT_FOLDER/${PREFIX}_group_gemm.csv
batch_gemm_out=$OUTPUT_FOLDER/${PREFIX}_batch_gemm.csv

# 定义颜色  
GREEN='\033[0;32m'  
BLUE='\033[0;34m'  
YELLOW='\033[1;33m'  
NC='\033[0m' # No Color  
BOLD='\033[1m' 

# 定义输出格式函数  
print_header() {  
    echo -e "\n${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}"  
    echo -e "${BOLD}${BLUE}  $1${NC}"  
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}\n"  
}  

print_step() {  
    echo -e "${YELLOW}➜ $1${NC}"  
}  

print_success() {  
    echo -e "${GREEN}✓ $1${NC}\n"  
}  

# FlashMLA  
print_header "Profiling FlashMLA"  
print_step "Running FlashMLA profiling..."  
python python/test_flash_mla.py | parse_mla_result_to_csv $mla_out  
print_success "FlashMLA results dumped to: $mla_out"  

# DeepGemm  
print_header "Profiling DeepGemm"  
print_step "Running DeepGemm profiling..."  
python python/test_decode_gemms.py --output-dir $OUTPUT_FOLDER --prefix ${PREFIX}_  
print_success "DeepGemm results dumped to:\n  ├─ $dense_gemm_out\n  └─ $group_gemm_out\n └─ $batch_gemm_out"  

# Process final result  
print_header "Processing Output Tables"  
print_step "Generating final results..."  
python python/process_table.py --dense_gemm $dense_gemm_out \
                       --group_gemm $group_gemm_out \
                       --batch_gemm $batch_gemm_out \
                       --mla $mla_out \
                       --output_path $OUTPUT_FOLDER \
                       --output_prefix ${PREFIX}-  

print_success "Final results generated at:\n  ├─ $OUTPUT_FOLDER/${PREFIX}-two-microbatch-overlapping.csv\n  └─ $OUTPUT_FOLDER/${PREFIX}-single-batch-comp-comm-overlapping.csv"  

print_header "Process Completed Successfully"
