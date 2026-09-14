`nvEncodeAPI.h` is unmodified from NVIDIA's MIT-licensed API headers, distributed
by https://github.com/FFmpeg/nv-codec-headers at tag `n12.2.72.0`:
https://github.com/FFmpeg/nv-codec-headers/blob/n12.2.72.0/include/ffnvcodec/nvEncodeAPI.h
The copyright and license are retained in the header. No FFmpeg implementation is included.

The adapter loads the NVIDIA driver at runtime. API 12.2 is used for compatibility
with current GB10 and RTX 5090 drivers. Supported codecs and formats are queried
on the active CUDA device rather than inferred from compute capability.
