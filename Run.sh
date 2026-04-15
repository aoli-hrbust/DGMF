#!/bin/bash

# 'Backbone-main'
Pro_file="DGMF_ablation"

ROOT_DIR="/PRML1501/mdh/Multi-view-task/${Pro_file}"
cd "$ROOT_DIR"

dir1="exp/logs"
dir2="exp/logs/P_ID.txt"
current_time=$(date +"%Y%m%d_%H%M%S")
dir1_log="${dir1}/print_${current_time}.log"

if [ ! -d "$dir1" ]; then
    mkdir -p "$dir1"
fi

cmd="./main-semi-classification.py"

#if [ "$Pro_file" == "GraphFusion-main" ] || [ "$Pro_file" == "JFGCN-main" ] || [ "$Pro_file" == "TrustGMM-main" ] ; then
#    cmd="./train.py"
#elif [ "$Pro_file" == "Backbone-main" ] || [ "$Pro_file" == "IMvGCN-main" ] || [ "$Pro_file" == "DCAP-re-main" ] ; then
#    cmd="./main.py"
#fi

# 'BBCnews', 'BBCsports', 'NGs', 'Citeseer', '3sources', 'MSRC-v1', 'ALOI',
# 'animals', 'Out_Scene', '100leaves', 'HW', 'MNIST', 'GRAZ02', 'Youtube',
# 'MNIST10k', 'Reuters', 'Wikipedia', 'NoisyMNIST_15000'
# 'BDGP', 'Citeseer', 'Cora', 'HW_2Views'
# 'YaleB_Extended'
# 'BRCA', 'ROSMAP', 'LGG', 'KIPAN'
# 'food101', 'MVSA_Single'

# clustering: 'CVPR23_GCFAgg', 'TMM25_SSLNMVC'
# classification: 'ICLR21_TMC', 'TPAMI22_ETMC', 'AAAI22_TMDLOA', 'IF24_RMVC', 'ICML25_TMCEK'
# 'AAAI24_ECML', 'ICML23_QMF_mm', 'AAAI25_TUNED'

#        --dataset=hw6\
# DGMF, wo_gmm, wo_fusion, wo_all, True, False
config="
        DGMF\
        --isConfig\
        --dir_h=main\
        --train_detail_dir=${ROOT_DIR}/${dir1_log}\
       "

cmd="${cmd} ${config}"

echo "${cmd}"
echo "${ROOT_DIR}/${dir1_log}"

nohup python -u $cmd >> $dir1_log 2>&1 &

PID=$!
date +"%Y-%m-%d %H:%M:%S" >> $dir2
echo "PID:"$PID" CMD:"$cmd"--->"$ROOT_DIR/$dir1_log >> $dir2
#tail -f $dir1_log