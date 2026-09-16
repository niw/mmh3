import Foundation
import Metal
import MetalPerformanceShaders

// Rust keeps each context on one thread. Command buffers retain their inputs until completion.
// Host reads and foreign-queue exports synchronize; batches are bounded by work and allocation.
private final class Context {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let tensorSource: String
    var tensorLibrary: MTLLibrary?
    var pipelines: [String: MTLComputePipelineState] = [:]
    var pending: MTLCommandBuffer?
    var operations = 0
    var allocatedSinceSync = 0
    var products: [String: MPSMatrixMultiplication] = [:]
    var reusable: [MTLBuffer] = []
    var retired: [MTLBuffer] = []
    var pooledBytes = 0
    var submissions: UInt64 = 0
    var allocations: UInt64 = 0
    var reuses: UInt64 = 0
    var peakBytes: UInt64 = 0
    var executionError: Error?
    let name: UnsafeMutablePointer<CChar>

    init(source: String, tensorSource: String) throws {
        self.tensorSource = tensorSource
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw BridgeError.message("No Metal GPU is available")
        }

        self.device = device
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe
        library = try device.makeLibrary(source: source, options: options)
        name = strdup(device.name)!
    }

    func command() throws -> MTLCommandBuffer {
        if let executionError {
            throw executionError
        }

        if let pending {
            return pending
        }

        guard let command = queue.makeCommandBuffer() else {
            throw BridgeError.message("No Metal command buffer")
        }

        pending = command
        return command
    }

    func synchronize() throws {
        if let executionError {
            throw executionError
        }

        guard let command = pending else {
            return
        }

        pending = nil
        operations = 0
        allocatedSinceSync = 0
        submissions += 1

        do {
            try complete(command)
        } catch {
            executionError = error
            throw error
        }

        reusable.append(contentsOf: retired)
        retired.removeAll(keepingCapacity: true)
    }

    func encoded() throws {
        operations += 1
        if operations >= 64 {
            try synchronize()
        }
    }

    func recycle(_ buffer: MTLBuffer) {
        // Never overwrite shared storage while the GPU can still be reading it.
        guard pooledBytes + buffer.length <= 64 * 1024 * 1024,
              reusable.count + retired.count < 128, executionError == nil
        else {
            return
        }

        pooledBytes += buffer.length
        if pending == nil {
            reusable.append(buffer)
        } else {
            retired.append(buffer)
        }
    }

    deinit {
        // Complete outstanding work even when unwinding an earlier Rust error.
        try? synchronize()
        free(name)
    }
}

private enum BridgeError: Error {
    case message(String)
}

private final class ErrorMessage {
    let pointer: UnsafeMutablePointer<CChar>

    init(_ message: String) {
        pointer = strdup(message)!
    }

    deinit {
        free(pointer)
    }
}

private func fail(_ error: Error) -> Int32 {
    Thread.current.threadDictionary["mmh3.metal.error"] = ErrorMessage(String(describing: error))
    return -1
}

private func context(_ pointer: UnsafeMutableRawPointer) -> Context {
    Unmanaged<Context>.fromOpaque(pointer).takeUnretainedValue()
}

private func buffer(_ pointer: UnsafeMutableRawPointer) -> MTLBuffer {
    Unmanaged<AnyObject>.fromOpaque(pointer).takeUnretainedValue() as! MTLBuffer
}

private func complete(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        throw command.error ?? BridgeError.message("Metal command failed")
    }
}

@_cdecl("mmh3_metal_error")
func metalError() -> UnsafePointer<CChar>? {
    (Thread.current.threadDictionary["mmh3.metal.error"] as? ErrorMessage).map {
        UnsafePointer($0.pointer)
    }
}

@_cdecl("mmh3_metal_create")
func metalCreate(_ source: UnsafePointer<CChar>, _ tensorSource: UnsafePointer<CChar>)
    -> UnsafeMutableRawPointer?
{
    autoreleasepool {
        do {
            return try Unmanaged.passRetained(
                Context(source: String(cString: source), tensorSource: String(cString: tensorSource))
            )
            .toOpaque()
        } catch {
            _ = fail(error)
            return nil
        }
    }
}

@_cdecl("mmh3_metal_destroy")
func metalDestroy(_ pointer: UnsafeMutableRawPointer) {
    Unmanaged<Context>.fromOpaque(pointer).release()
}

@_cdecl("mmh3_metal_name")
func metalName(_ pointer: UnsafeMutableRawPointer) -> UnsafePointer<CChar> {
    UnsafePointer(context(pointer).name)
}

@_cdecl("mmh3_metal_synchronize")
func metalSynchronize(_ pointer: UnsafeMutableRawPointer) -> Int32 {
    autoreleasepool {
        do {
            try context(pointer).synchronize()
            return 0
        } catch {
            return fail(error)
        }
    }
}

@_cdecl("mmh3_metal_stats")
func metalStats(_ pointer: UnsafeMutableRawPointer, _ output: UnsafeMutablePointer<UInt64>) {
    let ctx = context(pointer)
    output[0] = ctx.submissions
    output[1] = ctx.allocations
    output[2] = ctx.reuses
    output[3] = ctx.peakBytes
}

@_cdecl("mmh3_metal_alloc")
func metalAlloc(_ pointer: UnsafeMutableRawPointer, _ bytes: Int, _ data: UnsafeRawPointer?)
    -> UnsafeMutableRawPointer?
{
    autoreleasepool {
        let ctx = context(pointer)
        do {
            if let error = ctx.executionError {
                throw error
            }

            if ctx.pending != nil, ctx.allocatedSinceSync + bytes > 256 * 1024 * 1024 {
                try ctx.synchronize()
            }
        } catch {
            _ = fail(error)
            return nil
        }

        let result: MTLBuffer
        if let index = ctx.reusable.lastIndex(where: { $0.length == bytes }) {
            result = ctx.reusable.remove(at: index)
            ctx.pooledBytes -= bytes
            ctx.reuses += 1
        } else {
            guard let allocated = ctx.device.makeBuffer(length: bytes, options: .storageModeShared) else {
                _ = fail(BridgeError.message("Metal buffer allocation failed (\(bytes) bytes)"))
                return nil
            }

            result = allocated
            ctx.allocations += 1
            ctx.peakBytes = max(ctx.peakBytes, UInt64(ctx.device.currentAllocatedSize))
        }

        ctx.allocatedSinceSync += bytes
        if let data {
            result.contents().copyMemory(from: data, byteCount: bytes)
        }

        return Unmanaged.passRetained(result as AnyObject).toOpaque()
    }
}

@_cdecl("mmh3_metal_free")
func metalFree(_ ctx: UnsafeMutableRawPointer, _ pointer: UnsafeMutableRawPointer) {
    let allocation = Unmanaged<AnyObject>.fromOpaque(pointer).takeRetainedValue() as! MTLBuffer
    context(ctx).recycle(allocation)
}

@_cdecl("mmh3_metal_read")
func metalRead(_ pointer: UnsafeMutableRawPointer, _ output: UnsafeMutableRawPointer, _ bytes: Int) {
    output.copyMemory(from: buffer(pointer).contents(), byteCount: bytes)
}

@_cdecl("mmh3_metal_dispatch")
func metalDispatch(
    _ pointer: UnsafeMutableRawPointer, _ name: UnsafePointer<CChar>,
    _ buffers: UnsafePointer<UnsafeMutableRawPointer>, _ count: Int,
    _ parameters: UnsafeRawPointer, _ bytes: Int, _ threads: Int, _ groupSize: Int, _ groups: Bool
) -> Int32 {
    autoreleasepool {
        do {
            let ctx = context(pointer)
            let key = String(cString: name)
            let pipeline: MTLComputePipelineState
            if let cached = ctx.pipelines[key] {
                pipeline = cached
            } else {
                var library = ctx.library
                if key.hasPrefix("mpp_") {
                    guard #available(macOS 26.0, *), ctx.device.supportsFamily(.apple7) else {
                        throw BridgeError.message("MPP TensorOps requires macOS 26 and an Apple silicon GPU")
                    }

                    if ctx.tensorLibrary == nil {
                        let options = MTLCompileOptions()
                        options.languageVersion = .version4_0
                        options.mathMode = .safe
                        ctx.tensorLibrary = try ctx.device.makeLibrary(
                            source: ctx.tensorSource, options: options
                        )
                    }

                    library = ctx.tensorLibrary!
                }

                guard let function = library.makeFunction(name: key) else {
                    throw BridgeError.message("Missing Metal kernel: \(key)")
                }

                pipeline = try ctx.device.makeComputePipelineState(function: function)
                ctx.pipelines[key] = pipeline
            }

            guard
                !(key.hasPrefix("attention_") || key.hasPrefix("mpp_"))
                || pipeline.threadExecutionWidth == 32
            else {
                throw BridgeError.message("Tiled attention requires 32-thread SIMD groups")
            }

            let command = try ctx.command()
            guard groupSize <= pipeline.maxTotalThreadsPerThreadgroup,
                  let encoder = command.makeComputeCommandEncoder()
            else {
                throw BridgeError.message("Could not create Metal compute command")
            }

            encoder.setComputePipelineState(pipeline)
            for i in 0 ..< count {
                encoder.setBuffer(buffer(buffers[i]), offset: 0, index: i)
            }

            encoder.setBytes(parameters, length: bytes, index: count)
            let grid = MTLSize(width: threads, height: 1, depth: 1)
            let group = MTLSize(width: groupSize, height: 1, depth: 1)
            if groups {
                encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
            } else {
                encoder.dispatchThreads(grid, threadsPerThreadgroup: group)
            }

            encoder.endEncoding()
            try ctx.encoded()
            return 0
        } catch {
            return fail(error)
        }
    }
}

@_cdecl("mmh3_metal_matmul")
func metalMatmul(
    _ pointer: UnsafeMutableRawPointer, _ a: UnsafeMutableRawPointer,
    _ b: UnsafeMutableRawPointer, _ c: UnsafeMutableRawPointer, _ m: Int, _ n: Int, _ k: Int,
    _ stride: Int, _ offset: Int, _ halfInputs: Bool
) -> Int32 {
    autoreleasepool {
        do {
            let ctx = context(pointer)
            let elementBytes = halfInputs ? 2 : 4
            let inputType: MPSDataType = halfInputs ? .float16 : .float32
            let left = MPSMatrix(
                buffer: buffer(a),
                descriptor: MPSMatrixDescriptor(
                    rows: m, columns: k, rowBytes: k * elementBytes, dataType: inputType
                )
            )
            let right = MPSMatrix(
                buffer: buffer(b),
                descriptor: MPSMatrixDescriptor(
                    rows: n, columns: k, rowBytes: k * elementBytes, dataType: inputType
                )
            )
            let result = MPSMatrix(
                buffer: buffer(c), offset: offset * 4,
                descriptor: MPSMatrixDescriptor(
                    rows: m, columns: n, rowBytes: stride * 4, dataType: .float32
                )
            )
            let key = "\(m)/\(n)/\(k)/\(halfInputs)"
            let multiply: MPSMatrixMultiplication

            if let cached = ctx.products[key] {
                multiply = cached
            } else {
                multiply = MPSMatrixMultiplication(
                    device: ctx.device, transposeLeft: false, transposeRight: true,
                    resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1, beta: 0
                )
                ctx.products[key] = multiply
            }

            let command = try ctx.command()
            multiply.encode(
                commandBuffer: command, leftMatrix: left, rightMatrix: right, resultMatrix: result
            )
            try ctx.encoded()
            return 0
        } catch {
            return fail(error)
        }
    }
}

@_cdecl("mmh3_metal_supports_tensor_ops")
func metalSupportsTensorOps(_ pointer: UnsafeMutableRawPointer) -> Bool {
    if #available(macOS 26.0, *) {
        return context(pointer).device.supportsFamily(.apple7)
    }

    return false
}
