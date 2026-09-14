#include "nvEncodeAPI.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace {
struct Library {
    void *handle = nullptr;
    explicit Library(const char *name) {
#ifdef _WIN32
        handle = LoadLibraryA(name);
#else
        handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle)
            throw std::runtime_error(std::string("cannot load ") + name);
    }
    void *symbol(const char *name) {
#ifdef _WIN32
        auto address = reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
        auto address = dlsym(handle, name);
#endif
        if (!address)
            throw std::runtime_error(std::string("missing driver function ") + name);
        return address;
    }
    ~Library() {
#ifdef _WIN32
        if (handle)
            FreeLibrary(static_cast<HMODULE>(handle));
#else
        if (handle)
            dlclose(handle);
#endif
    }
};
struct Encoder {
#ifdef _WIN32
    Library cuda{"nvcuda.dll"};
    Library driver{"nvEncodeAPI64.dll"};
#else
    Library cuda{"libcuda.so.1"};
    Library driver{"libnvidia-encode.so.1"};
#endif
    NV_ENCODE_API_FUNCTION_LIST api{};
    void *session = nullptr;
    NV_ENC_REGISTERED_PTR registered = nullptr;
    NV_ENC_INPUT_PTR mapped = nullptr;
    NV_ENC_OUTPUT_PTR output = nullptr;
    bool locked = false;
    uint32_t width, height, pitch;
    std::vector<uint8_t> headers;
    Encoder(uint32_t width, uint32_t height, uint32_t pitch)
        : width(width), height(height), pitch(pitch) {}
    void check(NVENCSTATUS status, const char *operation) {
        if (status == NV_ENC_SUCCESS)
            return;
        std::string error =
            std::string(operation) + " failed (NVENC " + std::to_string(status) + ")";
        if (session && api.nvEncGetLastErrorString) {
            if (const char *detail = api.nvEncGetLastErrorString(session))
                error += std::string(": ") + detail;
        }
        throw std::runtime_error(error);
    }
    void release() {
        if (locked) {
            api.nvEncUnlockBitstream(session, output);
            locked = false;
        }
        if (mapped) {
            api.nvEncUnmapInputResource(session, mapped);
            mapped = nullptr;
        }
    }
    ~Encoder() {
        if (!session)
            return;
        release();
        if (registered)
            api.nvEncUnregisterResource(session, registered);
        if (output)
            api.nvEncDestroyBitstreamBuffer(session, output);
        api.nvEncDestroyEncoder(session);
    }
};
void message(char *destination, size_t size, const std::exception &exception) {
    if (size)
        std::snprintf(destination, size, "%s", exception.what());
}
bool same_guid(const GUID &left, const GUID &right) {
    return std::memcmp(&left, &right, sizeof(GUID)) == 0;
}
} // namespace

extern "C" {
struct Mmh3Packet {
    const uint8_t *data;
    uint32_t size;
    uint32_t keyframe;
    uint64_t pts;
};

void *mmh3_nvenc_open(void *input, uint32_t width, uint32_t height, uint32_t pitch, uint32_t fps,
                      char *error, size_t error_size) {
    try {
        auto encoder = std::make_unique<Encoder>(width, height, pitch);
        auto create = reinterpret_cast<NVENCSTATUS(NVENCAPI *)(NV_ENCODE_API_FUNCTION_LIST *)>(
            encoder->driver.symbol("NvEncodeAPICreateInstance"));
        encoder->api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        encoder->check(create(&encoder->api),
                       "NvEncodeAPICreateInstance (driver must support API 12.2)");
        void *context = nullptr;
        auto current =
            reinterpret_cast<int(NVENCAPI *)(void **)>(encoder->cuda.symbol("cuCtxGetCurrent"));
        if (current(&context) != 0 || !context)
            throw std::runtime_error("no active CUDA context");
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open{};
        open.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        open.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        open.device = context;
        open.apiVersion = NVENCAPI_VERSION;
        encoder->check(encoder->api.nvEncOpenEncodeSessionEx(&open, &encoder->session),
                       "NvEncOpenEncodeSessionEx");
        uint32_t count = 0;
        encoder->check(encoder->api.nvEncGetEncodeGUIDCount(encoder->session, &count),
                       "NvEncGetEncodeGUIDCount");
        std::vector<GUID> codecs(count);
        encoder->check(
            encoder->api.nvEncGetEncodeGUIDs(encoder->session, codecs.data(), count, &count),
            "NvEncGetEncodeGUIDs");
        if (std::none_of(codecs.begin(), codecs.end(),
                         [](const GUID &guid) { return same_guid(guid, NV_ENC_CODEC_H264_GUID); }))
            throw std::runtime_error("this CUDA device has no H.264 NVENC encoder");
        encoder->check(
            encoder->api.nvEncGetInputFormatCount(encoder->session, NV_ENC_CODEC_H264_GUID, &count),
            "NvEncGetInputFormatCount");
        std::vector<NV_ENC_BUFFER_FORMAT> formats(count);
        encoder->check(encoder->api.nvEncGetInputFormats(encoder->session, NV_ENC_CODEC_H264_GUID,
                                                         formats.data(), count, &count),
                       "NvEncGetInputFormats");
        if (std::find(formats.begin(), formats.end(), NV_ENC_BUFFER_FORMAT_NV12) == formats.end())
            throw std::runtime_error("this NVENC encoder does not accept NV12");
        for (auto dimension : {std::make_pair(NV_ENC_CAPS_WIDTH_MAX, width),
                               std::make_pair(NV_ENC_CAPS_HEIGHT_MAX, height)}) {
            NV_ENC_CAPS_PARAM caps{};
            caps.version = NV_ENC_CAPS_PARAM_VER;
            caps.capsToQuery = dimension.first;
            int limit = 0;
            encoder->check(encoder->api.nvEncGetEncodeCaps(encoder->session, NV_ENC_CODEC_H264_GUID,
                                                           &caps, &limit),
                           "NvEncGetEncodeCaps");
            if (dimension.second > static_cast<uint32_t>(limit))
                throw std::runtime_error("video dimensions exceed this NVENC encoder's limit");
        }
        NV_ENC_PRESET_CONFIG preset{};
        preset.version = NV_ENC_PRESET_CONFIG_VER;
        preset.presetCfg.version = NV_ENC_CONFIG_VER;
        encoder->check(encoder->api.nvEncGetEncodePresetConfigEx(
                           encoder->session, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P4_GUID,
                           NV_ENC_TUNING_INFO_HIGH_QUALITY, &preset),
                       "NvEncGetEncodePresetConfigEx");
        auto config = preset.presetCfg;
        config.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
        config.gopLength = fps * 2;
        config.frameIntervalP = 1; // No B frames: one synchronous output per input, DTS == PTS.
        config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
        config.rcParams.constQP = {20, 20, 20};
        config.rcParams.enableLookahead = 0;
        config.rcParams.lookaheadDepth = 0;
        auto &h264 = config.encodeCodecConfig.h264Config;
        h264.idrPeriod = fps * 2;
        h264.repeatSPSPPS = 1;
        h264.h264VUIParameters.videoSignalTypePresentFlag = 1;
        h264.h264VUIParameters.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
        h264.h264VUIParameters.videoFullRangeFlag = 0;
        h264.h264VUIParameters.colourDescriptionPresentFlag = 1;
        h264.h264VUIParameters.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_BT709;
        h264.h264VUIParameters.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
        h264.h264VUIParameters.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS_BT709;
        h264.h264VUIParameters.chromaSampleLocationFlag = 1;
        h264.h264VUIParameters.chromaSampleLocationTop = 1;
        h264.h264VUIParameters.chromaSampleLocationBot = 1;
        NV_ENC_INITIALIZE_PARAMS initialization{};
        initialization.version = NV_ENC_INITIALIZE_PARAMS_VER;
        initialization.encodeGUID = NV_ENC_CODEC_H264_GUID;
        initialization.presetGUID = NV_ENC_PRESET_P4_GUID;
        initialization.encodeWidth = width;
        initialization.encodeHeight = height;
        initialization.darWidth = width;
        initialization.darHeight = height;
        initialization.frameRateNum = fps;
        initialization.frameRateDen = 1;
        initialization.enablePTD = 1;
        initialization.encodeConfig = &config;
        initialization.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
        encoder->check(encoder->api.nvEncInitializeEncoder(encoder->session, &initialization),
                       "NvEncInitializeEncoder");
        NV_ENC_REGISTER_RESOURCE resource{};
        resource.version = NV_ENC_REGISTER_RESOURCE_VER;
        resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        resource.resourceToRegister = input;
        resource.width = width;
        resource.height = height;
        resource.pitch = pitch;
        resource.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
        resource.bufferUsage = NV_ENC_INPUT_IMAGE;
        encoder->check(encoder->api.nvEncRegisterResource(encoder->session, &resource),
                       "NvEncRegisterResource");
        encoder->registered = resource.registeredResource;
        NV_ENC_CREATE_BITSTREAM_BUFFER bitstream{};
        bitstream.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        encoder->check(encoder->api.nvEncCreateBitstreamBuffer(encoder->session, &bitstream),
                       "NvEncCreateBitstreamBuffer");
        encoder->output = bitstream.bitstreamBuffer;
        encoder->headers.resize(65536);
        uint32_t size = 0;
        NV_ENC_SEQUENCE_PARAM_PAYLOAD sequence{};
        sequence.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
        sequence.spsppsBuffer = encoder->headers.data();
        sequence.inBufferSize = static_cast<uint32_t>(encoder->headers.size());
        sequence.outSPSPPSPayloadSize = &size;
        encoder->check(encoder->api.nvEncGetSequenceParams(encoder->session, &sequence),
                       "NvEncGetSequenceParams");
        encoder->headers.resize(size);
        return encoder.release();
    } catch (const std::exception &exception) {
        message(error, error_size, exception);
        return nullptr;
    }
}
void mmh3_nvenc_headers(void *handle, const uint8_t **data, uint32_t *size) {
    auto *encoder = static_cast<Encoder *>(handle);
    *data = encoder->headers.data();
    *size = static_cast<uint32_t>(encoder->headers.size());
}
int mmh3_nvenc_encode(void *handle, uint64_t pts, Mmh3Packet *packet, char *error,
                      size_t error_size) {
    auto *encoder = static_cast<Encoder *>(handle);
    try {
        NV_ENC_MAP_INPUT_RESOURCE map{};
        map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
        map.registeredResource = encoder->registered;
        encoder->check(encoder->api.nvEncMapInputResource(encoder->session, &map),
                       "NvEncMapInputResource");
        encoder->mapped = map.mappedResource;
        NV_ENC_PIC_PARAMS picture{};
        picture.version = NV_ENC_PIC_PARAMS_VER;
        picture.inputBuffer = encoder->mapped;
        picture.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
        picture.inputWidth = encoder->width;
        picture.inputHeight = encoder->height;
        picture.inputPitch = encoder->pitch;
        picture.outputBitstream = encoder->output;
        picture.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
        picture.inputTimeStamp = pts;
        picture.inputDuration = 1;
        encoder->check(encoder->api.nvEncEncodePicture(encoder->session, &picture),
                       "NvEncEncodePicture");
        NV_ENC_LOCK_BITSTREAM lock{};
        lock.version = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream = encoder->output;
        encoder->check(encoder->api.nvEncLockBitstream(encoder->session, &lock),
                       "NvEncLockBitstream");
        encoder->locked = true;
        *packet = {static_cast<const uint8_t *>(lock.bitstreamBufferPtr), lock.bitstreamSizeInBytes,
                   lock.pictureType == NV_ENC_PIC_TYPE_IDR ? 1u : 0u, lock.outputTimeStamp};
        return 0;
    } catch (const std::exception &exception) {
        encoder->release();
        message(error, error_size, exception);
        return -1;
    }
}
void mmh3_nvenc_release(void *handle) { static_cast<Encoder *>(handle)->release(); }
int mmh3_nvenc_finish(void *handle, char *error, size_t error_size) {
    auto *encoder = static_cast<Encoder *>(handle);
    try {
        NV_ENC_PIC_PARAMS picture{};
        picture.version = NV_ENC_PIC_PARAMS_VER;
        picture.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
        encoder->check(encoder->api.nvEncEncodePicture(encoder->session, &picture),
                       "NvEncEncodePicture(EOS)");
        return 0;
    } catch (const std::exception &exception) {
        message(error, error_size, exception);
        return -1;
    }
}
void mmh3_nvenc_close(void *handle) { delete static_cast<Encoder *>(handle); }
}
