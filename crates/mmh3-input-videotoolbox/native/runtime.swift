import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

// VideoToolbox behind a decoder that mirrors the NVDEC adapter: Rust hands it parameter sets and
// access units with four-byte NAL lengths, and takes every frame back as NV12 in host memory, at
// the size the stream crops its pictures to, with the display position it came in with.

private enum DecoderError: Error { case message(String) }

private func check(_ status: OSStatus, _ operation: String) throws {
    if status != noErr {
        throw DecoderError.message("\(operation) failed (OSStatus \(status))")
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
    Thread.current.threadDictionary["mmh3.videotoolbox.decoder.error"] = ErrorMessage(
        String(describing: error)
    )
}

// The NAL units of data whose units each follow their four-byte length.
private func units(_ data: UnsafeBufferPointer<UInt8>) throws -> [UnsafeBufferPointer<UInt8>] {
    var units: [UnsafeBufferPointer<UInt8>] = []
    var offset = 0
    while offset + 4 <= data.count {
        let length = data[offset ..< offset + 4].reduce(0) { $0 << 8 | Int($1) }
        offset += 4
        guard length > 0, offset + length <= data.count else {
            throw DecoderError.message("A NAL unit runs past its data")
        }

        units.append(UnsafeBufferPointer(rebasing: data[offset ..< offset + length]))
        offset += length
    }
    return units
}

// The code of H.264's VUI for the matrix a format description names, or 2 when it names none.
private func matrixCode(_ format: CMFormatDescription) -> Int32 {
    let matrix = CMFormatDescriptionGetExtension(
        format, extensionKey: kCMFormatDescriptionExtension_YCbCrMatrix
    )
    guard let matrix, CFGetTypeID(matrix) == CFStringGetTypeID() else {
        return 2
    }

    switch matrix as! CFString {
    case kCVImageBufferYCbCrMatrix_ITU_R_709_2:
        return 1
    case kCVImageBufferYCbCrMatrix_ITU_R_601_4:
        return 6
    case kCVImageBufferYCbCrMatrix_SMPTE_240M_1995:
        return 7
    case kCVImageBufferYCbCrMatrix_ITU_R_2020:
        return 9
    default:
        return 2
    }
}

// Rust owns this object on one thread. The VideoToolbox callback only touches the locked queue.
private final class Decoder {
    let format: CMVideoFormatDescription
    let width: Int
    let height: Int
    let matrix: Int32
    let fullRange: Bool
    var session: VTDecompressionSession?
    private let lock = NSLock()
    // Decoded frames in the order they come out, which is their decoding order, with their
    // display positions.
    private var ready: [(position: Int, frame: Data)] = []
    private var callbackStatus: OSStatus = noErr

    init(parameterSets: UnsafeBufferPointer<UInt8>) throws {
        let sets = try units(parameterSets)
        // The format takes the sequence sets before the picture sets.
        let sequenceSets = sets.filter { $0.first.map { $0 & 0x1F } == 7 }
        let pictureSets = sets.filter { $0.first.map { $0 & 0x1F } == 8 }
        guard !sequenceSets.isEmpty, !pictureSets.isEmpty else {
            throw DecoderError.message("The parameter sets lack a sequence or picture set")
        }

        let ordered = sequenceSets + pictureSets

        var format: CMVideoFormatDescription?
        try check(
            CMVideoFormatDescriptionCreateFromH264ParameterSets(
                allocator: nil, parameterSetCount: ordered.count,
                parameterSetPointers: ordered.map { $0.baseAddress! },
                parameterSetSizes: ordered.map(\.count), nalUnitHeaderLength: 4,
                formatDescriptionOut: &format
            ),
            "reading H.264 parameter sets"
        )
        guard let format else {
            throw DecoderError.message("No H.264 format description")
        }

        // The dimensions are those the sequence set crops its pictures to.
        let dimensions = CMVideoFormatDescriptionGetDimensions(format)
        self.format = format
        width = Int(dimensions.width)
        height = Int(dimensions.height)
        matrix = matrixCode(format)
        let range = CMFormatDescriptionGetExtension(
            format, extensionKey: kCMFormatDescriptionExtension_FullRangeVideo
        )
        fullRange = (range as? Bool) ?? false
        guard width > 0, height > 0, width % 2 == 0, height % 2 == 0 else {
            throw DecoderError.message("Unsupported H.264 frame size \(width) x \(height)")
        }

        // The pixels come out in the range the stream holds them in, so none is converted.
        let attributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: fullRange
                ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        ]
        var callback = VTDecompressionOutputCallbackRecord(
            decompressionOutputCallback: { reference, frame, status, flags, image, _, _ in
                guard let reference else {
                    return
                }

                // The frame's reference is its display position plus one, so that none is null.
                let decoder = Unmanaged<Decoder>.fromOpaque(reference).takeUnretainedValue()
                decoder.receive(
                    position: Int(bitPattern: frame) - 1, status: status, flags: flags, image: image
                )
            }, decompressionOutputRefCon: Unmanaged.passUnretained(self).toOpaque()
        )
        try check(
            VTDecompressionSessionCreate(
                allocator: nil, formatDescription: format, decoderSpecification: nil,
                imageBufferAttributes: attributes as CFDictionary,
                outputCallback: &callback, decompressionSessionOut: &session
            ),
            "creating H.264 decoder"
        )
    }

    deinit {
        if let session {
            VTDecompressionSessionInvalidate(session)
        }
    }

    private func receive(
        position: Int, status: OSStatus, flags: VTDecodeInfoFlags, image: CVImageBuffer?
    ) {
        var frame: Data?
        var status = status
        if status == noErr, let image {
            do {
                frame = try copy(image)
            } catch {
                status = kVTVideoDecoderBadDataErr
            }
        } else if status == noErr, !flags.contains(.frameDropped) {
            status = kVTVideoDecoderBadDataErr
        }
        lock.lock()
        if status != noErr, callbackStatus == noErr {
            callbackStatus = status
        }
        if let frame {
            ready.append((position, frame))
        }
        lock.unlock()
    }

    // The frame's cropped pixels as NV12: the luma rows, then the interleaved chroma rows.
    private func copy(_ image: CVPixelBuffer) throws -> Data {
        try check(CVPixelBufferLockBaseAddress(image, .readOnly), "locking a decoded frame")
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        guard CVPixelBufferGetPlaneCount(image) == 2,
              CVPixelBufferGetWidthOfPlane(image, 0) >= width,
              CVPixelBufferGetHeightOfPlane(image, 0) >= height,
              let y = CVPixelBufferGetBaseAddressOfPlane(image, 0),
              let uv = CVPixelBufferGetBaseAddressOfPlane(image, 1)
        else {
            throw DecoderError.message("Expected an NV12 frame of \(width) x \(height)")
        }

        let yStride = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(image, 1)
        var frame = Data(count: width * height * 3 / 2)
        frame.withUnsafeMutableBytes { destination in
            let base = destination.baseAddress!
            for row in 0 ..< height {
                base.advanced(by: row * width).copyMemory(
                    from: y.advanced(by: row * yStride), byteCount: width
                )
            }
            for row in 0 ..< height / 2 {
                base.advanced(by: (height + row) * width).copyMemory(
                    from: uv.advanced(by: row * uvStride), byteCount: width
                )
            }
        }
        return frame
    }

    private func status() throws {
        lock.lock()
        let status = callbackStatus
        lock.unlock()
        try check(status, "decoding a frame")
    }

    func decode(_ unit: UnsafeBufferPointer<UInt8>, position: Int) throws {
        guard let session, let address = unit.baseAddress, unit.count > 0 else {
            throw DecoderError.message("No decoder or an empty access unit")
        }

        var block: CMBlockBuffer?
        try check(
            CMBlockBufferCreateWithMemoryBlock(
                allocator: nil, memoryBlock: nil, blockLength: unit.count, blockAllocator: nil,
                customBlockSource: nil, offsetToData: 0, dataLength: unit.count,
                flags: kCMBlockBufferAssureMemoryNowFlag, blockBufferOut: &block
            ),
            "allocating an access unit"
        )
        guard let block else {
            throw DecoderError.message("No block buffer")
        }

        try check(
            CMBlockBufferReplaceDataBytes(
                with: address, blockBuffer: block, offsetIntoDestination: 0, dataLength: unit.count
            ),
            "copying an access unit"
        )
        var sample: CMSampleBuffer?
        var size = unit.count
        try check(
            CMSampleBufferCreateReady(
                allocator: nil, dataBuffer: block, formatDescription: format, sampleCount: 1,
                sampleTimingEntryCount: 0, sampleTimingArray: nil, sampleSizeEntryCount: 1,
                sampleSizeArray: &size, sampleBufferOut: &sample
            ),
            "wrapping an access unit"
        )
        guard let sample else {
            throw DecoderError.message("No sample buffer")
        }

        // The session releases frames in decoding order even when asked for temporal processing,
        // so every frame carries its display position back for Rust to reorder by.
        try check(
            VTDecompressionSessionDecodeFrame(
                session, sampleBuffer: sample, flags: [],
                frameRefcon: UnsafeMutableRawPointer(bitPattern: position + 1), infoFlagsOut: nil
            ),
            "decoding an access unit"
        )
        try status()
    }

    // Ends the stream so the decoder releases every frame it still holds.
    func flush() throws {
        guard let session else {
            throw DecoderError.message("No decoder")
        }

        try check(VTDecompressionSessionFinishDelayedFrames(session), "finishing the stream")
        try check(VTDecompressionSessionWaitForAsynchronousFrames(session), "waiting for frames")
        try status()
    }

    var readyCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return ready.count
    }

    // Copies the oldest frame out, returning its display position.
    func take(_ destination: UnsafeMutableRawBufferPointer) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        guard let (position, frame) = ready.first, destination.count >= frame.count else {
            return nil
        }

        frame.copyBytes(to: destination.bindMemory(to: UInt8.self))
        ready.removeFirst()
        return position
    }
}

@_cdecl("mmh3_vtdec_error")
func decoderError() -> UnsafePointer<CChar>? {
    (Thread.current.threadDictionary["mmh3.videotoolbox.decoder.error"] as? ErrorMessage).map {
        UnsafePointer($0.pointer)
    }
}

@_cdecl("mmh3_vtdec_create")
func decoderCreate(
    _ parameterSets: UnsafePointer<UInt8>, _ length: Int
) -> UnsafeMutableRawPointer? {
    autoreleasepool {
        do {
            let sets = UnsafeBufferPointer(start: parameterSets, count: length)
            return try Unmanaged.passRetained(Decoder(parameterSets: sets)).toOpaque()
        } catch {
            fail(error)
            return nil
        }
    }
}

// The size of the frames with the colour matrix, by H.264's code, and range of the stream.
@_cdecl("mmh3_vtdec_size")
func decoderSize(
    _ pointer: UnsafeMutableRawPointer, _ width: UnsafeMutablePointer<Int32>,
    _ height: UnsafeMutablePointer<Int32>, _ matrix: UnsafeMutablePointer<Int32>,
    _ fullRange: UnsafeMutablePointer<Int32>
) {
    let decoder = Unmanaged<Decoder>.fromOpaque(pointer).takeUnretainedValue()
    width.pointee = Int32(decoder.width)
    height.pointee = Int32(decoder.height)
    matrix.pointee = decoder.matrix
    fullRange.pointee = decoder.fullRange ? 1 : 0
}

// Decodes one access unit of the frame shown at `position`, queueing the frame it completes.
@_cdecl("mmh3_vtdec_feed")
func decoderFeed(
    _ pointer: UnsafeMutableRawPointer, _ data: UnsafePointer<UInt8>, _ length: Int, _ position: Int
) -> Int32 {
    autoreleasepool {
        do {
            try Unmanaged<Decoder>.fromOpaque(pointer).takeUnretainedValue().decode(
                UnsafeBufferPointer(start: data, count: length), position: position
            )
            return 0
        } catch {
            fail(error)
            return -1
        }
    }
}

@_cdecl("mmh3_vtdec_flush")
func decoderFlush(_ pointer: UnsafeMutableRawPointer) -> Int32 {
    autoreleasepool {
        do {
            try Unmanaged<Decoder>.fromOpaque(pointer).takeUnretainedValue().flush()
            return 0
        } catch {
            fail(error)
            return -1
        }
    }
}

@_cdecl("mmh3_vtdec_ready")
func decoderReady(_ pointer: UnsafeMutableRawPointer) -> Int {
    Unmanaged<Decoder>.fromOpaque(pointer).takeUnretainedValue().readyCount
}

// Copies the oldest decoded frame into `nv12`, the luma plane then the interleaved chroma plane,
// and its display position into `position`.
@_cdecl("mmh3_vtdec_take")
func decoderTake(
    _ pointer: UnsafeMutableRawPointer, _ nv12: UnsafeMutablePointer<UInt8>, _ capacity: Int,
    _ position: UnsafeMutablePointer<Int>
) -> Int32 {
    let decoder = Unmanaged<Decoder>.fromOpaque(pointer).takeUnretainedValue()
    let destination = UnsafeMutableRawBufferPointer(start: nv12, count: capacity)
    guard let taken = decoder.take(destination) else {
        return -1
    }

    position.pointee = taken
    return 0
}

@_cdecl("mmh3_vtdec_destroy")
func decoderDestroy(_ pointer: UnsafeMutableRawPointer) {
    Unmanaged<Decoder>.fromOpaque(pointer).release()
}
