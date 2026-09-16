import CoreMedia
import CoreVideo
import Foundation
import Metal
import VideoToolbox

private enum EncoderError: Error { case message(String) }

private func check(_ status: OSStatus, _ operation: String) throws {
    if status != noErr {
        throw EncoderError.message("\(operation) failed (OSStatus \(status))")
    }
}

private final class ErrorMessage {
    let pointer: UnsafeMutablePointer<CChar>
    init(_ message: String) {
        pointer = strdup(message)!
    }

    deinit { free(pointer) }
}

private func fail(_ error: Error) {
    Thread.current.threadDictionary["mmh3.videotoolbox.error"] = ErrorMessage(
        String(describing: error)
    )
}

// Rust owns this object on one thread. Only the VideoToolbox callback accesses the locked result.
private final class Encoder {
    let width: Int
    let height: Int
    let fps: Int32
    var session: VTCompressionSession?
    var converter: MetalConverter?
    private let lock = NSLock()
    private var received: CMSampleBuffer?
    private var callbackStatus: OSStatus = noErr
    var packet = NSData()
    var sps = NSData()
    var pps = NSData()
    var keyframe = false

    init(width: Int32, height: Int32, fps: Int32) throws {
        self.width = Int(width)
        self.height = Int(height)
        self.fps = fps
        let attributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true,
        ]
        try check(
            VTCompressionSessionCreate(
                allocator: nil, width: width, height: height, codecType: kCMVideoCodecType_H264,
                encoderSpecification: [
                    kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder: true
                ] as CFDictionary,
                imageBufferAttributes: attributes as CFDictionary, compressedDataAllocator: nil,
                outputCallback: { reference, _, status, _, sample in
                    guard let reference else {
                        return
                    }

                    let encoder = Unmanaged<Encoder>.fromOpaque(reference).takeUnretainedValue()
                    encoder.lock.lock()
                    encoder.callbackStatus = status
                    encoder.received = sample
                    encoder.lock.unlock()
                }, refcon: Unmanaged.passUnretained(self).toOpaque(), compressionSessionOut: &session
            ),
            "creating H.264 hardware encoder"
        )

        guard let session else {
            throw EncoderError.message("No VideoToolbox session")
        }

        let properties: [CFString: Any] = [
            kVTCompressionPropertyKey_AllowFrameReordering: false,
            kVTCompressionPropertyKey_RealTime: false,
            kVTCompressionPropertyKey_ProfileLevel: kVTProfileLevel_H264_High_AutoLevel,
            kVTCompressionPropertyKey_ExpectedFrameRate: fps,
            kVTCompressionPropertyKey_MaxKeyFrameInterval: Int64(fps) * 2,
            kVTCompressionPropertyKey_Quality: 0.8,
            kVTCompressionPropertyKey_ColorPrimaries: kCVImageBufferColorPrimaries_ITU_R_709_2,
            kVTCompressionPropertyKey_TransferFunction: kCVImageBufferTransferFunction_ITU_R_709_2,
            kVTCompressionPropertyKey_YCbCrMatrix: kCVImageBufferYCbCrMatrix_ITU_R_709_2,
        ]
        try check(
            VTSessionSetProperties(session, propertyDictionary: properties as CFDictionary),
            "configuring encoder"
        )
        try check(VTCompressionSessionPrepareToEncodeFrames(session), "preparing encoder")
    }

    deinit {
        if let session {
            VTCompressionSessionInvalidate(session)
        }
    }

    func encode(_ source: UnsafePointer<UInt8>, index: Int64) throws {
        guard let session, let pool = VTCompressionSessionGetPixelBufferPool(session) else {
            throw EncoderError.message("No encoder pixel buffer pool")
        }

        var pixelBuffer: CVPixelBuffer?
        try check(CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer), "allocating video frame")
        guard let pixelBuffer else {
            throw EncoderError.message("No pixel buffer")
        }

        try check(CVPixelBufferLockBaseAddress(pixelBuffer, []), "locking video frame")
        do {
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
            guard CVPixelBufferGetPlaneCount(pixelBuffer) == 2,
                  let y = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
                  let uv = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1)
            else {
                throw EncoderError.message("Expected NV12 video frame")
            }

            let yStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
            let uvStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)
            for row in 0 ..< height {
                y.advanced(by: row * yStride).copyMemory(
                    from: source.advanced(by: row * width), byteCount: width
                )
            }

            let u = source.advanced(by: width * height)
            let v = u.advanced(by: width * height / 4)
            for row in 0 ..< height / 2 {
                let destination = uv.advanced(by: row * uvStride).assumingMemoryBound(to: UInt8.self)
                for column in 0 ..< width / 2 {
                    let i = row * (width / 2) + column
                    destination[column * 2] = u[i]
                    destination[column * 2 + 1] = v[i]
                }
            }
        }

        try encode(pixelBuffer, index: index)
    }

    func encodeMetal(_ buffer: MTLBuffer, frames: Int, sourceIndex: Int64, index: Int64) throws {
        guard let session, let pool = VTCompressionSessionGetPixelBufferPool(session), let converter
        else {
            throw EncoderError.message("No Metal encoder pixel buffer pool")
        }

        var pixelBuffer: CVPixelBuffer?
        try check(
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer), "allocating shared video frame"
        )
        guard let pixelBuffer else {
            throw EncoderError.message("No shared video frame")
        }

        try converter.convert(
            buffer, to: pixelBuffer, width: width, height: height, frames: frames, index: sourceIndex
        )
        try encode(pixelBuffer, index: index)
    }

    private func encode(_ pixelBuffer: CVPixelBuffer, index: Int64) throws {
        guard let session else {
            throw EncoderError.message("No video encoder")
        }

        CVBufferSetAttachment(
            pixelBuffer, kCVImageBufferChromaLocationTopFieldKey, kCVImageBufferChromaLocation_Center,
            .shouldPropagate
        )
        CVBufferSetAttachment(
            pixelBuffer, kCVImageBufferChromaLocationBottomFieldKey, kCVImageBufferChromaLocation_Center,
            .shouldPropagate
        )
        lock.lock()
        received = nil
        callbackStatus = noErr
        lock.unlock()
        let pts = CMTime(value: index, timescale: fps)
        try check(
            VTCompressionSessionEncodeFrame(
                session, imageBuffer: pixelBuffer, presentationTimeStamp: pts,
                duration: CMTime(value: 1, timescale: fps), frameProperties: nil, sourceFrameRefcon: nil,
                infoFlagsOut: nil
            ),
            "encoding video frame"
        )
        // CompleteFrames waits for the callback before Rust can reuse the input or read the packet.
        try check(
            VTCompressionSessionCompleteFrames(session, untilPresentationTimeStamp: .invalid),
            "completing video frame"
        )
        lock.lock()
        let sample = received
        let status = callbackStatus
        lock.unlock()
        try check(status, "video callback")
        guard let sample, CMSampleBufferGetNumSamples(sample) == 1,
              CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(sample), pts) == 0,
              let format = CMSampleBufferGetFormatDescription(sample),
              let data = CMSampleBufferGetDataBuffer(sample)
        else {
            throw EncoderError.message("Encoder did not return the expected frame")
        }

        var parameters: [NSData] = []
        for i in 0 ..< 2 {
            var pointer: UnsafePointer<UInt8>?
            var size = 0
            var nalLength: Int32 = 0
            try check(
                CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                    format, parameterSetIndex: i,
                    parameterSetPointerOut: &pointer, parameterSetSizeOut: &size, parameterSetCountOut: nil,
                    nalUnitHeaderLengthOut: &nalLength
                ), "reading H.264 configuration"
            )

            guard let pointer, size > 0, nalLength == 4 else {
                throw EncoderError.message("Expected four-byte H.264 NAL lengths")
            }

            parameters.append(NSData(bytes: pointer, length: size))
        }

        sps = parameters[0]
        pps = parameters[1]
        let length = CMBlockBufferGetDataLength(data)
        guard length > 0, let bytes = NSMutableData(length: length) else {
            throw EncoderError.message("Empty H.264 packet")
        }

        try check(
            CMBlockBufferCopyDataBytes(
                data, atOffset: 0, dataLength: length, destination: bytes.mutableBytes
            ),
            "copying H.264 packet"
        )
        packet = bytes
        let attachments =
            CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: false)
                as? [[CFString: Any]]
        keyframe = attachments?.first?[kCMSampleAttachmentKey_NotSync] as? Bool != true
    }
}

// CVMetalTexture views share the encoder's IOSurface allocation. They remain alive until the
// conversion command has completed, then VideoToolbox reads that same pixel buffer.
private final class MetalConverter {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    let cache: CVMetalTextureCache

    init(source: String) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EncoderError.message("No Metal device for video conversion")
        }

        self.device = device
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: source, options: options)

        guard let function = library.makeFunction(name: "rgb_to_nv12") else {
            throw EncoderError.message("No RGB to NV12 kernel")
        }

        pipeline = try device.makeComputePipelineState(function: function)
        var cache: CVMetalTextureCache?
        try check(
            CVMetalTextureCacheCreate(nil, nil, device, nil, &cache), "creating video texture cache"
        )

        guard let cache else {
            throw EncoderError.message("No video texture cache")
        }

        self.cache = cache
    }

    func convert(
        _ input: MTLBuffer, to pixelBuffer: CVPixelBuffer, width: Int, height: Int, frames: Int,
        index: Int64
    ) throws {
        guard input.device.registryID == device.registryID,
              input.length >= 3 * frames * width * height * 4,
              index >= 0, index < frames
        else {
            throw EncoderError.message("RGB buffer does not match the Metal output device or dimensions")
        }

        let attributes = [kCVMetalTextureUsage: MTLTextureUsage.shaderWrite.rawValue] as CFDictionary
        var y: CVMetalTexture?
        var uv: CVMetalTexture?
        try check(
            CVMetalTextureCacheCreateTextureFromImage(
                nil, cache, pixelBuffer, attributes, .r8Unorm, width, height, 0, &y
            ), "mapping NV12 luma"
        )
        try check(
            CVMetalTextureCacheCreateTextureFromImage(
                nil, cache, pixelBuffer, attributes, .rg8Unorm, width / 2, height / 2, 1, &uv
            ),
            "mapping NV12 chroma"
        )
        guard let y, let uv, let yTexture = CVMetalTextureGetTexture(y),
              let uvTexture = CVMetalTextureGetTexture(uv),
              let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
        else {
            throw EncoderError.message("Could not create shared NV12 textures or conversion command")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        let parameters = [UInt32(width), UInt32(height), UInt32(frames), UInt32(index)]
        parameters.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 1) }
        encoder.setTexture(yTexture, index: 0)
        encoder.setTexture(uvTexture, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: width / 2, height: height / 2, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        withExtendedLifetime((y, uv, input, pixelBuffer)) {
        }

        guard command.status == .completed else {
            throw command.error ?? EncoderError.message("Metal video conversion failed")
        }
    }
}

@_cdecl("mmh3_vt_enable_metal")
func encoderEnableMetal(_ pointer: UnsafeMutableRawPointer, _ source: UnsafePointer<CChar>) -> Int32 {
    autoreleasepool {
        do {
            Unmanaged<Encoder>.fromOpaque(pointer).takeUnretainedValue().converter = try MetalConverter(
                source: String(cString: source)
            )
            return 0
        } catch {
            fail(error)
            return -1
        }
    }
}

@_cdecl("mmh3_vt_encode_metal")
func encoderEncodeMetal(
    _ pointer: UnsafeMutableRawPointer, _ buffer: UnsafeMutableRawPointer, _ frames: Int,
    _ sourceIndex: Int64, _ index: Int64
) -> Int32 {
    autoreleasepool {
        do {
            let input = Unmanaged<AnyObject>.fromOpaque(buffer).takeUnretainedValue() as! MTLBuffer
            try Unmanaged<Encoder>.fromOpaque(pointer).takeUnretainedValue().encodeMetal(
                input, frames: frames, sourceIndex: sourceIndex, index: index
            )
            return 0
        } catch {
            fail(error)
            return -1
        }
    }
}

@_cdecl("mmh3_vt_error")
func encoderError() -> UnsafePointer<CChar>? {
    (Thread.current.threadDictionary["mmh3.videotoolbox.error"] as? ErrorMessage).map {
        UnsafePointer($0.pointer)
    }
}

@_cdecl("mmh3_vt_create")
func encoderCreate(_ width: Int32, _ height: Int32, _ fps: Int32) -> UnsafeMutableRawPointer? {
    autoreleasepool {
        do {
            return try Unmanaged.passRetained(Encoder(width: width, height: height, fps: fps)).toOpaque()
        } catch {
            fail(error)
            return nil
        }
    }
}

@_cdecl("mmh3_vt_destroy")
func encoderDestroy(_ pointer: UnsafeMutableRawPointer) {
    Unmanaged<Encoder>.fromOpaque(pointer).release()
}

@_cdecl("mmh3_vt_encode")
func encoderEncode(
    _ pointer: UnsafeMutableRawPointer, _ source: UnsafePointer<UInt8>, _ index: Int64
) -> Int32 {
    autoreleasepool {
        do {
            try Unmanaged<Encoder>.fromOpaque(pointer).takeUnretainedValue().encode(source, index: index)
            return 0
        } catch {
            fail(error)
            return -1
        }
    }
}

// Packet and parameter-set storage remains owned by Encoder until its next encode call.
@_cdecl("mmh3_vt_bytes")
func encoderBytes(
    _ pointer: UnsafeMutableRawPointer, _ kind: Int32, _ count: UnsafeMutablePointer<Int>
) -> UnsafeRawPointer {
    let encoder = Unmanaged<Encoder>.fromOpaque(pointer).takeUnretainedValue()
    let bytes = kind == 0 ? encoder.packet : (kind == 1 ? encoder.sps : encoder.pps)
    count.pointee = bytes.length
    return bytes.bytes
}

@_cdecl("mmh3_vt_keyframe")
func encoderKeyframe(_ pointer: UnsafeMutableRawPointer) -> Bool {
    Unmanaged<Encoder>.fromOpaque(pointer).takeUnretainedValue().keyframe
}
