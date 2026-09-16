#include "dynlink_cuda.h"
#include "dynlink_nvcuvid.h"
#include <cstdio>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef _WIN32
#include <dlfcn.h>
#endif

// NVDEC through the driver, loaded at run time: an H.264 parser feeds a decoder, and every frame
// the parser sends to display is copied to the host as NV12. The pixels become RGB on the CPU,
// where the copy is a third of the size it would be as floats.

namespace {

struct Library {
    void *handle = nullptr;
    explicit Library(const char *name) {
#ifdef _WIN32
        handle = LoadLibraryA(name);
#else
        handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle) {
            throw std::runtime_error(std::string("cannot load ") + name);
        }
    }
    void *symbol(const char *name) {
#ifdef _WIN32
        auto address = reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
        auto address = dlsym(handle, name);
#endif
        if (!address) {
            throw std::runtime_error(std::string("missing driver function ") + name);
        }
        return address;
    }
    ~Library() {
#ifdef _WIN32
        if (handle)
            FreeLibrary(static_cast<HMODULE>(handle));
#else
        if (handle) {
            dlclose(handle);
        }
#endif
    }
    Library(const Library &) = delete;
    Library &operator=(const Library &) = delete;
};

struct Decoder {
#ifdef _WIN32
    Library cuda{"nvcuda.dll"};
    Library nvcuvid{"nvcuvid.dll"};
#else
    Library cuda{"libcuda.so.1"};
    Library nvcuvid{"libnvcuvid.so.1"};
#endif
    tcuInit *Init = nullptr;
    tcuCtxGetCurrent *CtxGetCurrent = nullptr;
    tcuCtxPushCurrent_v2 *CtxPushCurrent = nullptr;
    tcuDeviceGet *DeviceGet = nullptr;
    tcuDevicePrimaryCtxRetain *PrimaryCtxRetain = nullptr;
    tcuMemcpy2D_v2 *Memcpy2D = nullptr;
    tcuvidCreateVideoParser *CreateVideoParser = nullptr;
    tcuvidParseVideoData *ParseVideoData = nullptr;
    tcuvidDestroyVideoParser *DestroyVideoParser = nullptr;
    tcuvidCreateDecoder *CreateDecoder = nullptr;
    tcuvidDestroyDecoder *DestroyDecoder = nullptr;
    tcuvidDecodePicture *DecodePicture = nullptr;
    tcuvidMapVideoFrame64 *MapVideoFrame = nullptr;
    tcuvidUnmapVideoFrame64 *UnmapVideoFrame = nullptr;

    CUvideoparser parser = nullptr;
    CUvideodecoder decoder = nullptr;
    unsigned width = 0;
    unsigned height = 0;
    // The colour matrix and range the stream says its pixels carry, 2 when it says nothing.
    int matrix = 2;
    int full_range = 0;
    std::deque<std::vector<uint8_t>> ready;
    std::string failure;

    Decoder() {
        Init = reinterpret_cast<tcuInit *>(cuda.symbol("cuInit"));
        CtxGetCurrent = reinterpret_cast<tcuCtxGetCurrent *>(cuda.symbol("cuCtxGetCurrent"));
        CtxPushCurrent =
            reinterpret_cast<tcuCtxPushCurrent_v2 *>(cuda.symbol("cuCtxPushCurrent_v2"));
        DeviceGet = reinterpret_cast<tcuDeviceGet *>(cuda.symbol("cuDeviceGet"));
        PrimaryCtxRetain =
            reinterpret_cast<tcuDevicePrimaryCtxRetain *>(cuda.symbol("cuDevicePrimaryCtxRetain"));
        Memcpy2D = reinterpret_cast<tcuMemcpy2D_v2 *>(cuda.symbol("cuMemcpy2D_v2"));
        CreateVideoParser =
            reinterpret_cast<tcuvidCreateVideoParser *>(nvcuvid.symbol("cuvidCreateVideoParser"));
        ParseVideoData =
            reinterpret_cast<tcuvidParseVideoData *>(nvcuvid.symbol("cuvidParseVideoData"));
        DestroyVideoParser =
            reinterpret_cast<tcuvidDestroyVideoParser *>(nvcuvid.symbol("cuvidDestroyVideoParser"));
        CreateDecoder =
            reinterpret_cast<tcuvidCreateDecoder *>(nvcuvid.symbol("cuvidCreateDecoder"));
        DestroyDecoder =
            reinterpret_cast<tcuvidDestroyDecoder *>(nvcuvid.symbol("cuvidDestroyDecoder"));
        DecodePicture =
            reinterpret_cast<tcuvidDecodePicture *>(nvcuvid.symbol("cuvidDecodePicture"));
        MapVideoFrame =
            reinterpret_cast<tcuvidMapVideoFrame64 *>(nvcuvid.symbol("cuvidMapVideoFrame64"));
        UnmapVideoFrame =
            reinterpret_cast<tcuvidUnmapVideoFrame64 *>(nvcuvid.symbol("cuvidUnmapVideoFrame64"));
    }

    ~Decoder() {
        if (parser) {
            DestroyVideoParser(parser);
        }
        if (decoder) {
            DestroyDecoder(decoder);
        }
    }

    /// The decoder needs a context; the process usually has the runtime's already.
    void adopt_context() {
        CUcontext context = nullptr;
        if (CtxGetCurrent(&context) == CUDA_SUCCESS && context != nullptr) {
            return;
        }
        CUdevice device = 0;
        if (Init(0) != CUDA_SUCCESS || DeviceGet(&device, 0) != CUDA_SUCCESS ||
            PrimaryCtxRetain(&context, device) != CUDA_SUCCESS ||
            CtxPushCurrent(context) != CUDA_SUCCESS) {
            throw std::runtime_error("cannot make a CUDA context current");
        }
    }
};

int CUDAAPI on_sequence(void *user, CUVIDEOFORMAT *format) {
    auto *decoder = static_cast<Decoder *>(user);
    if (format->bit_depth_luma_minus8 != 0 || format->chroma_format != cudaVideoChromaFormat_420) {
        decoder->failure = "only 8-bit 4:2:0 video is supported";
        return 0;
    }
    const unsigned width = format->display_area.right - format->display_area.left;
    const unsigned height = format->display_area.bottom - format->display_area.top;
    if (decoder->decoder != nullptr) {
        // A second sequence header of the same stream needs no new decoder.
        return (width == decoder->width && height == decoder->height)
                   ? static_cast<int>(format->min_num_decode_surfaces)
                   : 0;
    }
    CUVIDDECODECREATEINFO info{};
    info.CodecType = format->codec;
    info.ChromaFormat = format->chroma_format;
    info.OutputFormat = cudaVideoSurfaceFormat_NV12;
    info.bitDepthMinus8 = format->bit_depth_luma_minus8;
    info.DeinterlaceMode = format->progressive_sequence ? cudaVideoDeinterlaceMode_Weave
                                                        : cudaVideoDeinterlaceMode_Adaptive;
    info.ulNumOutputSurfaces = 2;
    info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    info.ulNumDecodeSurfaces = format->min_num_decode_surfaces;
    info.ulWidth = format->coded_width;
    info.ulHeight = format->coded_height;
    info.ulMaxWidth = format->coded_width;
    info.ulMaxHeight = format->coded_height;
    info.display_area.left = static_cast<short>(format->display_area.left);
    info.display_area.top = static_cast<short>(format->display_area.top);
    info.display_area.right = static_cast<short>(format->display_area.right);
    info.display_area.bottom = static_cast<short>(format->display_area.bottom);
    info.ulTargetWidth = width;
    info.ulTargetHeight = height;
    if (decoder->CreateDecoder(&decoder->decoder, &info) != CUDA_SUCCESS) {
        decoder->failure = "the driver refused the decoder";
        return 0;
    }
    decoder->width = width;
    decoder->height = height;
    decoder->matrix = format->video_signal_description.matrix_coefficients;
    decoder->full_range = format->video_signal_description.video_full_range_flag;
    return static_cast<int>(format->min_num_decode_surfaces);
}

int CUDAAPI on_decode(void *user, CUVIDPICPARAMS *parameters) {
    auto *decoder = static_cast<Decoder *>(user);
    if (decoder->decoder == nullptr) {
        return 0;
    }
    if (decoder->DecodePicture(decoder->decoder, parameters) != CUDA_SUCCESS) {
        decoder->failure = "a picture failed to decode";
        return 0;
    }
    return 1;
}

// Frames arrive here in display order, which is the order the clip is encoded in.
int CUDAAPI on_display(void *user, CUVIDPARSERDISPINFO *info) {
    auto *decoder = static_cast<Decoder *>(user);
    if (decoder->decoder == nullptr || info == nullptr) {
        return 0;
    }
    CUVIDPROCPARAMS parameters{};
    parameters.progressive_frame = info->progressive_frame;
    parameters.top_field_first = info->top_field_first;
    parameters.unpaired_field = info->repeat_first_field < 0;
    unsigned long long frame = 0;
    unsigned pitch = 0;
    if (decoder->MapVideoFrame(decoder->decoder, info->picture_index, &frame, &pitch,
                               &parameters) != CUDA_SUCCESS) {
        decoder->failure = "a decoded frame could not be mapped";
        return 0;
    }
    const unsigned width = decoder->width;
    const unsigned height = decoder->height;
    std::vector<uint8_t> nv12(static_cast<size_t>(width) * height * 3 / 2);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = static_cast<CUdeviceptr>(frame);
    copy.srcPitch = pitch;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = nv12.data();
    copy.dstPitch = width;
    copy.WidthInBytes = width;
    copy.Height = height;
    CUresult status = decoder->Memcpy2D(&copy);
    if (status == CUDA_SUCCESS) {
        // The interleaved chroma plane follows the luma rows of the surface.
        copy.srcDevice = static_cast<CUdeviceptr>(frame + static_cast<size_t>(pitch) * height);
        copy.dstHost = nv12.data() + static_cast<size_t>(width) * height;
        copy.Height = height / 2;
        status = decoder->Memcpy2D(&copy);
    }
    decoder->UnmapVideoFrame(decoder->decoder, frame);
    if (status != CUDA_SUCCESS) {
        decoder->failure = "a decoded frame could not be copied";
        return 0;
    }
    decoder->ready.push_back(std::move(nv12));
    return 1;
}

int feed(Decoder *decoder, const uint8_t *data, size_t length, bool end) {
    CUVIDSOURCEDATAPACKET packet{};
    packet.payload = data;
    packet.payload_size = static_cast<unsigned long>(length);
    packet.flags = CUVID_PKT_TIMESTAMP;
    if (end) {
        packet.flags |= CUVID_PKT_ENDOFSTREAM;
    }
    if (decoder->ParseVideoData(decoder->parser, &packet) != CUDA_SUCCESS) {
        if (decoder->failure.empty()) {
            decoder->failure = "the parser rejected a packet";
        }
        return 1;
    }
    return decoder->failure.empty() ? 0 : 1;
}

} // namespace

// Creates a parser for an H.264 track and hands it the track's Annex B parameter sets. The
// decoder itself waits for the first access unit, which is when the parser reports the format.
// Returns 0 on success.
extern "C" int mmh3_nvdec_create(const uint8_t *parameter_sets, size_t length, void **handle) {
    Decoder *decoder = nullptr;
    try {
        decoder = new Decoder();
        decoder->adopt_context();
        CUVIDPARSERPARAMS parameters{};
        parameters.CodecType = cudaVideoCodec_H264;
        parameters.ulMaxNumDecodeSurfaces = 20;
        // Frames come out in display order, so the parser may hold a few back.
        parameters.ulMaxDisplayDelay = 4;
        parameters.pUserData = decoder;
        parameters.pfnSequenceCallback = on_sequence;
        parameters.pfnDecodePicture = on_decode;
        parameters.pfnDisplayPicture = on_display;
        if (decoder->CreateVideoParser(&decoder->parser, &parameters) != CUDA_SUCCESS) {
            throw std::runtime_error("the driver refused a video parser");
        }
        if (feed(decoder, parameter_sets, length, false) != 0) {
            throw std::runtime_error(decoder->failure.empty() ? "the parameter sets are not H.264"
                                                              : decoder->failure);
        }
    } catch (const std::exception &error) {
        fprintf(stderr, "nvdec: %s\n", error.what());
        delete decoder;
        *handle = nullptr;
        return 1;
    }
    *handle = decoder;
    return 0;
}

// The size the frames come out at, once the parser has reported the format, with the colour matrix
// and range of the stream. Returns 0 when they are known.
extern "C" int mmh3_nvdec_size(void *handle, int *width, int *height, int *matrix,
                               int *full_range) {
    auto *decoder = static_cast<Decoder *>(handle);
    if (decoder->decoder == nullptr) {
        return 1;
    }
    *width = static_cast<int>(decoder->width);
    *height = static_cast<int>(decoder->height);
    *matrix = decoder->matrix;
    *full_range = decoder->full_range;
    return 0;
}

// Parses one Annex B access unit, queueing whatever frames it completes.
extern "C" int mmh3_nvdec_feed(void *handle, const uint8_t *data, size_t length) {
    return feed(static_cast<Decoder *>(handle), data, length, false);
}

// Ends the stream so the parser sends every frame it still holds to display.
extern "C" int mmh3_nvdec_flush(void *handle) {
    return feed(static_cast<Decoder *>(handle), nullptr, 0, true);
}

extern "C" size_t mmh3_nvdec_ready(void *handle) {
    return static_cast<Decoder *>(handle)->ready.size();
}

// Copies the oldest queued frame into `nv12`, the luma plane then the interleaved chroma plane.
extern "C" int mmh3_nvdec_take(void *handle, uint8_t *nv12, size_t capacity) {
    auto *decoder = static_cast<Decoder *>(handle);
    if (decoder->ready.empty()) {
        return 1;
    }
    const std::vector<uint8_t> &frame = decoder->ready.front();
    if (capacity < frame.size()) {
        return 1;
    }
    std::memcpy(nv12, frame.data(), frame.size());
    decoder->ready.pop_front();
    return 0;
}

extern "C" const char *mmh3_nvdec_error(void *handle) {
    auto *decoder = static_cast<Decoder *>(handle);
    return decoder->failure.empty() ? nullptr : decoder->failure.c_str();
}

extern "C" void mmh3_nvdec_destroy(void *handle) { delete static_cast<Decoder *>(handle); }
