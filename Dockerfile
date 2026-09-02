# Qwen3.8-Flash-Next (qwen4_exp) NVFP4 on RTX PRO 6000 (sm_120).
# The model family ships only inside the official day-0 image below, and that
# image's sglang is a source install of /sgl-workspace/sglang, so the build
# patches that tree. Every patch must apply cleanly or the build fails.
#
#   0001  fp8_e4m3 KV cache end to end (fp8 tile dequant in the QSA prefill kernels, exact chunked-prefill rereads, FlashInfer paged decode on SM120)
#   0002  GDN on SM120 (fp32 initial state + int64 cu_seqlens, RecoverSSM via --gdn-mtp-cache-mode=none and the WY output-only kernel, matching KV budget)
#   0003  online MXFP8 for the unquantized linears behind SGLANG_SM120_ONLINE_MXFP8, plus fp8 weight replacement (HyperConnection mix pair born meta and ingested rowwise fp8 from CPU shards, lm_head replaced at end of load, no bf16 fallback, reader asserts fire)
#   0004  serve the local-inference-lab release as-is (config-class aliases, vision_config dict-to-class, nvfp4-packed PLE shards dequantized at load)

FROM lmsysorg/sglang:qwen38flashnext@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae

WORKDIR /sgl-workspace/sglang

COPY patches/ /opt/qwen38-patches/

# Apply the patches; the build stops if any patch does not apply cleanly to the pinned base image.
RUN set -eux; \
    cd /sgl-workspace/sglang; \
    for p in $(ls /opt/qwen38-patches/*.patch | sort); do \
        echo "=== applying $(basename $p) ==="; \
        git apply --check "$p" || { echo "ERROR: $(basename $p) does not apply cleanly to the image tree"; exit 1; }; \
        git apply "$p"; \
    done; \
    rm -rf /opt/qwen38-patches

# Check that all patches applied successfully and modifications can be loaded.
# No GPU is needed. The flashinfer WY kernel import additionally checks the cutlass DSL and cuda-python companions.
RUN python3 -c "import inspect, pathlib; from sglang.kernels.ops.gemm import sm120_online_fp8 as o; import sglang.srt.layers.attention.linear.kernels.gdn_flashinfer as g; assert callable(o.sm120_fp8_lm_head_logits) and callable(o.replace_linears_with_fp8_copies) and callable(o.rowwise_scale_of) and callable(o.dequant_rowwise_weight), 'the fp8 weight-replacement primitives are absent'; assert callable(o.rowwise_mix_enabled) and callable(o.attach_rowwise_ingest) and 'device=_mix_device' in pathlib.Path('python/sglang/srt/layers/hyperconnection.py').read_text(), 'the at-load ingest rail is absent'; assert 'self.post_load_weights()' in pathlib.Path('python/sglang/srt/models/qwen4_exp.py').read_text(), 'the post-load hook is not wired into load_weights'; assert '_log_weight_dtype_census' in pathlib.Path('python/sglang/srt/models/qwen4_exp.py').read_text(), 'the dtype census is absent'; assert hasattr(g, 'is_flashinfer_gdn_wy_output_only_available'), 'patch 0002 did not apply'; from flashinfer.gdn_kernels import gated_delta_rule_mtp_wy_output_only as wy; assert wy is not None and callable(wy), 'the flashinfer WY kernel module must be importable (is nvidia-cutlass-dsl installed?)'; assert callable(o.maybe_sm120_fp8_lm_head), 'patch 0003 did not apply'; assert 'online_fp8_enabled' in pathlib.Path('python/sglang/srt/layers/hc_mix_triton.py').read_text(), 'patch 0003 did not repoint the HyperConnection mix'; import sglang.srt.server_args as sa; assert hasattr(sa.ServerArgs, 'gdn_mtp_cache_mode'), 'patch 0002 did not add its server argument'; from sglang.srt.layers.attention.qwen_sparse_attn_backend import _TRTLLM_SPARSE_PAGE_SIZE; assert _TRTLLM_SPARSE_PAGE_SIZE > 0, 'patch 0001 did not apply'; import sglang.srt.mem_cache.kv_cache_configurator as k; assert 'no_intermediate_ssm' in inspect.getsource(k.KVCacheConfigurator._handle_max_mamba_cache), 'patch 0002 did not apply'; assert '_maybe_convert_linears_to_mxfp8' in pathlib.Path('python/sglang/srt/models/qwen4_exp.py').read_text(), 'patch 0003 did not add the linear conversion hook'; assert 'SGLANG_SM120_ONLINE_MXFP8' in pathlib.Path('python/sglang/srt/environ.py').read_text(), 'patch 0003 did not add its environment switch'; assert 'Qwen3_8FlashNextForConditionalGeneration' in pathlib.Path('python/sglang/srt/configs/model_config.py').read_text(), 'patch 0004 did not normalize the architecture name'; assert 'qwen3_8_flash_next' in pathlib.Path('python/sglang/srt/utils/hf_transformers/common.py').read_text(), 'patch 0004 did not alias the config class'; assert 'unmatched checkpoint scales' in pathlib.Path('python/sglang/srt/models/qwen4_exp.py').read_text(), 'patch 0004 did not add the unmatched-scale reporting pass'; assert '_ple_source_packed' in pathlib.Path('python/sglang/srt/models/qwen4_exp.py').read_text(), 'patch 0004 did not add the packed PLE path'; assert 'sm120-turbo r20]' in pathlib.Path('python/sglang/srt/models/qwen4_exp.py').read_text(), 'the stack log prefix is absent'; assert 'the map alone decides' in pathlib.Path('python/sglang/srt/model_loader/weight_utils.py').read_text(), 'patch 0004 did not fix modelopt_mixed completeness'; assert 'isinstance(\n                text_config, Qwen4ExpTextConfig' in pathlib.Path('python/sglang/srt/configs/model_config.py').read_text(), 'patch 0004 did not guard the config rebuild'; print('patch sanity OK: all patches applied and their modifications load')"
