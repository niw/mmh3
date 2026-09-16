`dynlink_cuda.h`, `dynlink_cuviddec.h` and `dynlink_nvcuvid.h` are unmodified from NVIDIA's
MIT-licensed API headers, distributed by https://github.com/FFmpeg/nv-codec-headers at tag
`n12.2.72.0`:
https://github.com/FFmpeg/nv-codec-headers/tree/n12.2.72.0/include/ffnvcodec
The copyright and license are retained in the headers. No FFmpeg implementation is included.

The adapter loads the NVIDIA driver at run time, as the NVENC one does, and takes the
process's CUDA context when it has one. It decodes 8-bit 4:2:0 H.264 and copies each frame
to the host as NV12, which the crate turns into RGB.
