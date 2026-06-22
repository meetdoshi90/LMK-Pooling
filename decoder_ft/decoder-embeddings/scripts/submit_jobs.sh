
# bsub -q normal -n 1 -R "span[ptile=1]" -U infusion \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo lmk_mistral7bv3_finallayer_var_lmk.out \
#     -eo lmk_mistral7bv3_finallayer_var_lmk.err \
#     -J lmk_mistral7bv3_finallayer_var_lmk \
#     bash scripts/lmk_mistral7bv3_finallayer.sh


# bsub -q normal -n 1 -R "span[ptile=1]" \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo lmk_mean_mistral7bv3_finallayer.out \
#     -eo lmk_mean_mistral7bv3_finallayer.err \
#     -J lmk_mean_mistral7bv3_finallayer \
#     bash scripts/lmk_mean_eos_mistral7bv3_finallayer.sh mean


# bsub -q normal -n 1 -R "span[ptile=1]" \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo lmk_eos_mistral7bv3_finallayer.out \
#     -eo lmk_eos_mistral7bv3_finallayer.err \
#     -J lmk_eos_mistral7bv3_finallayer \
#     bash scripts/lmk_mean_eos_mistral7bv3_finallayer.sh last


# bsub -q normal -n 1 -R "span[ptile=1]" -U infusion \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo lmk_granite8_33_finallayer_var_lmk.out \
#     -eo lmk_granite8_33_finallayer_var_lmk.err \
#     -J lmk_granite8_33_finallayer_var_lmk \
#     bash scripts/lmk_finallayer.sh


# bsub -q normal -n 1 -R "span[ptile=1]" \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo lmk_mean_granite8_33_finallayer.out \
#     -eo lmk_mean_granite8_33_finallayer.err \
#     -J lmk_mean_granite8_33_finallayer \
#     bash scripts/mean_eos_finallayer.sh mean


# bsub -q normal -n 1 -R "span[ptile=1]" \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo lmk_eos_granite8_33_finallayer.out \
#     -eo lmk_eos_granite8_33_finallayer.err \
#     -J lmk_eos_granite8_33_finallayer \
#     bash scripts/mean_eos_finallayer.sh last



# bsub -q normal -n 1 -R "span[ptile=1]" -U infusion \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo msmarco_lmk_mistral7bv3_finallayer_var_lmk.out \
#     -eo msmarco_lmk_mistral7bv3_finallayer_var_lmk.err \
#     -J msmarco_lmk_mistral7bv3_finallayer_var_lmk \
#     bash scripts/msmarco_lmk_finallayer.sh


# bsub -q normal -n 1 -R "span[ptile=1]" \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo msmarco_lmk_mean_mistral7bv3_finallayer.out \
#     -eo msmarco_lmk_mean_mistral7bv3_finallayer.err \
#     -J msmarco_lmk_mean_mistral7bv3_finallayer \
#     bash scripts/msmarco_mean_eos_finallayer.sh mean


# bsub -q normal -n 1 -R "span[ptile=1]" \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo msmarco_lmk_eos_mistral7bv3_finallayer.out \
#     -eo msmarco_lmk_eos_mistral7bv3_finallayer.err \
#     -J msmarco_lmk_eos_mistral7bv3_finallayer \
#     bash scripts/msmarco_mean_eos_finallayer.sh last




# bsub -q normal -n 1 -R "span[ptile=1]" -U infusion \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo msmarco_lmk_granite418b_finallayer_var_lmk.out \
#     -eo msmarco_lmk_granite418b_finallayer_var_lmk.err \
#     -J msmarco_lmk_granite418b_finallayer_var_lmk \
#     bash scripts/msmarco_lmk_finallayer.sh


# bsub -q normal -n 1 -R "span[ptile=1]" \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo msmarco_lmk_mean_granite418b_finallayer.out \
#     -eo msmarco_lmk_mean_granite418b_finallayer.err \
#     -J msmarco_lmk_mean_granite418b_finallayer \
#     bash scripts/msmarco_mean_eos_finallayer.sh mean


# bsub -q normal -n 1 -R "span[ptile=1]" \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo msmarco_lmk_eos_granite418b_finallayer.out \
#     -eo msmarco_lmk_eos_granite418b_finallayer.err \
#     -J msmarco_lmk_eos_granite418b_finallayer \
#     bash scripts/msmarco_mean_eos_finallayer.sh last


# bsub -q normal -n 1 -R "span[ptile=1]" -U infusion \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo lmk_llm2vec_llama38b_inst_mntp_unsup_simcse_finallayer_var_lmk.out \
#     -eo lmk_llm2vec_llama38b_inst_mntp_unsup_simcse_finallayer_var_lmk.err \
#     -J lmk_llm2vec_llama38b_inst_mntp_unsup_simcse_finallayer_var_lmk \
#     bash scripts/lmk_finallayer.sh


bsub -q normal -n 1 -R "span[ptile=1]" \
    -U infusion \
    -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
    -M 900GB \
    -oo lmk_mean_llm2vec_llama38b_inst_mntp_unsup_simcse_finallayer.out \
    -eo lmk_mean_llm2vec_llama38b_inst_mntp_unsup_simcse_finallayer.err \
    -J lmk_mean_llm2vec_llama38b_inst_mntp_unsup_simcse_finallayer \
    bash scripts/mean_eos_finallayer.sh mean


# bsub -q normal -n 1 -R "span[ptile=1]" \
#     -U infusion \
#     -gpu "num=8/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:glink=yes" \
#     -M 900GB \
#     -oo lmk_eos_llm2vec_llama38b_inst_mntp_unsup_simcse_finallayer.out \
#     -eo lmk_eos_llm2vec_llama38b_inst_mntp_unsup_simcse_finallayer.err \
#     -J lmk_eos_llm2vec_llama38b_inst_mntp_unsup_simcse_finallayer \
#     bash scripts/mean_eos_finallayer.sh last
