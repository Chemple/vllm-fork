MAX_JOBS=32 \
CMAKE_BUILD_TYPE=Debug \
VERBOSE=1 \
CUDA_HOME=/usr/local/cuda-12.6 \
VLLM_FLASH_ATTN_VERSION=3 \
VLLM_TRACE_FUNCTION=1 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
VLLM_CPU_KVCACHE_SPACE=16 \
# use V1 version
VLLM_USE_V1=1 \
pip install -e .
