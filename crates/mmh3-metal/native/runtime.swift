import Foundation
import Metal
import MetalPerformanceShaders

// Bytes every context holds in its pool, which the memory a caller is told it may still fill
// counts as free: a pooled buffer is handed out again, or let go of when the device needs the room.
private let poolLock = NSLock()
private let weightsLabel = "weights"
private var pooledEverywhere = 0

private func pooled(_ change: Int) {
    poolLock.lock()
    pooledEverywhere += change
    poolLock.unlock()
}

// One context serves every thread of the process, one thread at a time: its lock is taken for the
// whole of every call that touches it, since its queue, its pipelines and its batch are one of
// each. Command buffers retain their inputs until completion. Host reads and foreign-queue exports
// synchronize; batches are bounded by work and allocation. A full batch is committed without
// waiting for it, so the GPU runs it while the next one is encoded, and at most `inFlightLimit`
// run at once.
private final class Context {
    let lock = NSLock()
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let tensorSource: String
    var tensorLibrary: MTLLibrary?
    lazy var fragmentLayout = probeFragments()
    var pipelines: [String: MTLComputePipelineState] = [:]
    var pending: MTLCommandBuffer?
    var operations = 0
    var allocatedSinceSync = 0
    var products: [String: MPSMatrixMultiplication] = [:]
    var reusable: [MTLBuffer] = []
    var retired: [MTLBuffer] = []
    /// Committed batches not known to be done, oldest first, each with the buffers let go of while
    /// it or one before it could still read them, which are free once it completes.
    var inFlight: [(command: MTLCommandBuffer, retired: [MTLBuffer])] = []
    let inFlightLimit = 2
    var pooledBytes = 0 {
        didSet { pooled(pooledBytes - oldValue) }
    }

    // The pool holds buffers up to half of what the device recommends one process fill, up to 8 GB.
    // A DiT block allocates the same few outputs of gigabytes each, and a fresh one costs its pages
    // on first touch, so the pool gains most when it holds all of them.
    let poolLimit: Int
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
        poolLimit = min(Int(device.recommendedMaxWorkingSetSize) / 2, 8 << 30)
        let options = MTLCompileOptions()
        options.mathMode = .safe
        library = try device.makeLibrary(source: source, options: options)
        name = strdup(device.name)!
    }

    /// Where this GPU's cooperative tensors put a lane's elements, found by running `fragmentProbe`
    /// once. The kernels in matmul.metal keep their fragments in registers in one of two layouts,
    /// and a GPU that uses neither runs none of them.
    func probeFragments() -> FragmentLayout {
        guard #available(macOS 26.0, *), device.supportsFamily(.apple7) else {
            return .unknown
        }

        do {
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: fragmentProbe, options: options)
            guard let function = library.makeFunction(name: "fragment_probe"),
                  let output = device.makeBuffer(
                      length: fragmentProbeValues * MemoryLayout<Int32>.stride,
                      options: .storageModeShared
                  ),
                  let commands = queue.makeCommandBuffer(),
                  let encoder = commands.makeComputeCommandEncoder()
            else {
                return .unknown
            }

            let pipeline = try device.makeComputePipelineState(function: function)
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(output, offset: 0, index: 0)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
            encoder.endEncoding()
            commands.commit()
            commands.waitUntilCompleted()
            guard commands.status == .completed else {
                return .unknown
            }

            let values = UnsafeBufferPointer(
                start: output.contents().bindMemory(to: Int32.self, capacity: fragmentProbeValues),
                count: fragmentProbeValues
            )
            return [FragmentLayout.consecutive, .pairs].first { $0.matches(values) } ?? .unknown
        } catch {
            return .unknown
        }
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

    /// Commits the batch being encoded, if any, without waiting for it, and waits for the oldest
    /// batches until no more than `limit` are in flight.
    func flush(leaving limit: Int) throws {
        if let executionError {
            throw executionError
        }

        if let command = pending {
            pending = nil
            operations = 0
            allocatedSinceSync = 0
            submissions += 1
            command.commit()
            inFlight.append((command, retired))
            retired.removeAll(keepingCapacity: true)
        }

        while inFlight.count > limit {
            let (command, freed) = inFlight.removeFirst()
            do {
                try complete(command)
            } catch {
                executionError = error
                throw error
            }

            reusable.append(contentsOf: freed)
        }
    }

    func synchronize() throws {
        try flush(leaving: 0)
        // Buffers let go of with nothing pending were read by nothing after the last batch.
        reusable.append(contentsOf: retired)
        retired.removeAll(keepingCapacity: true)
    }

    func encoded() throws {
        operations += 1
        if operations >= 64 {
            try flush(leaving: inFlightLimit)
        }
    }

    func recycle(_ buffer: MTLBuffer) {
        guard buffer.length <= poolLimit, buffer.label != weightsLabel, executionError == nil else {
            return
        }
        // The oldest free buffers make room in bytes and in number, so a text encoder's many small
        // outputs cannot crowd out the DiT's. Retired buffers stay, since the GPU may still read
        // them.
        while pooledBytes + buffer.length > poolLimit || reusable.count + retired.count >= 128,
              !reusable.isEmpty
        {
            pooledBytes -= reusable.removeFirst().length
        }
        guard pooledBytes + buffer.length <= poolLimit, reusable.count + retired.count < 128
        else {
            return
        }

        pooledBytes += buffer.length
        if pending == nil, inFlight.isEmpty {
            reusable.append(buffer)
        } else {
            retired.append(buffer)
        }
    }

    deinit {
        // Complete outstanding work even when unwinding an earlier Rust error.
        try? synchronize()
        pooled(-pooledBytes)
        free(name)
    }
}

/// Writes where the cooperative tensors of the 16 × 32 × 16 products in matmul.metal put each lane's
/// elements: for the left operand, the right one and the destination, with b transposed and not, in
/// FP16 and INT8, a lane's capacity and then each element's two indices.
private let fragmentProbe = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename T> void put(device int *out, ushort lane, thread T &tensor) {
    device int *o = out + lane * 33;
    o[0] = int(tensor.get_capacity());
    for (ushort i = 0; i < 16 && i < tensor.get_capacity(); ++i) {
        auto index = tensor.get_multidimensional_index(i);
        o[1 + i * 2] = int(index[0]);
        o[2 + i * 2] = int(index[1]);
    }
}

template <bool TRANSPOSE, typename A, typename B, typename C>
void probe(device int *out, ushort lane) {
    constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, TRANSPOSE, true,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
    auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
    auto destination = op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(left)>, remove_addrspace_t<decltype(right)>, C>();
    put(out, lane, left);
    put(out + 32 * 33, lane, right);
    put(out + 64 * 33, lane, destination);
}

kernel void fragment_probe(device int *out [[buffer(0)]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    probe<true, half, half, float>(out, lane);
    probe<false, half, half, float>(out + 96 * 33, lane);
    probe<true, int8_t, int8_t, int>(out + 192 * 33, lane);
    probe<false, int8_t, int8_t, int>(out + 288 * 33, lane);
}
"""

/// Four products of three tensors of 32 lanes, each a capacity and 16 pairs of indices.
private let fragmentProbeValues = 4 * 3 * 32 * 33

/// Where the kernels in matmul.metal expect the cooperative tensors to put a lane's elements, as
/// FRAGMENT_PAIRS there says.
enum FragmentLayout {
    /// Columns c to c + 3 of a lane's rows r and r + 8, as on GPUs with the Neural Accelerators.
    case consecutive
    /// Columns c, c + 1, c + 8 and c + 9, as on the GPUs before them.
    case pairs
    case unknown

    /// Whether `values`, which `fragmentProbe` wrote, put every element where this layout does.
    func matches(_ values: UnsafeBufferPointer<Int32>) -> Bool {
        for product in 0 ..< 4 {
            let transposed = product % 2 == 0
            for role in 0 ..< 3 {
                for lane in 0 ..< 32 {
                    let at = ((product * 3 + role) * 32 + lane) * 33
                    let capacity = role == 0 ? 8 : 16
                    guard values[at] == capacity else {
                        return false
                    }

                    for i in 0 ..< capacity {
                        let (x, y) = index(role: role, transposed: transposed, lane: lane, element: i)
                        guard values[at + 1 + i * 2] == x, values[at + 2 + i * 2] == y else {
                            return false
                        }
                    }
                }
            }
        }

        return true
    }

    /// The two indices get_multidimensional_index gives the element i of a lane in the left
    /// operand (role 0), the right one (1) or the destination (2), as matmul.metal's
    /// fragment_coord, fragment_column, fragment_of and element_of place it.
    private func index(role: Int, transposed: Bool, lane: Int, element i: Int) -> (Int32, Int32) {
        let pairs = self == .pairs
        let row = (lane & 16) >> 2 | (lane >> 1) & 3
        let column = pairs ? (lane & 1) * 2 + (lane >> 3 & 1) * 4 : (lane & 1) * 4 + (lane >> 3 & 1) * 8
        func place(_ e: Int) -> (x: Int, y: Int) {
            let j = e & 3
            return (column + (pairs ? (j & 1) + (j >> 1) * 8 : j), row + (e >> 2) * 8)
        }

        if role == 0 {
            let at = place(i)
            return (Int32(at.x), Int32(at.y))
        }

        let f = pairs ? (i >> 2) & 1 : i >> 3
        let at = place(pairs ? (i >> 3) << 2 | (i & 3) : i & 7)
        if role == 1, transposed {
            return pairs ? (Int32(at.y), Int32(at.x + 16 * f)) : (Int32(at.x), Int32(at.y + 16 * f))
        }

        return (Int32(at.x + 16 * f), Int32(at.y))
    }
}

private enum BridgeError: Error {
    case message(String)
}

private final class ErrorMessage {
    let pointer: UnsafeMutablePointer<CChar>
    let outOfMemory: Bool

    init(_ message: String, outOfMemory: Bool) {
        pointer = strdup(message)!
        self.outOfMemory = outOfMemory
    }

    deinit {
        free(pointer)
    }
}

private func fail(_ error: Error, outOfMemory: Bool = false) -> Int32 {
    Thread.current.threadDictionary["mmh3.metal.error"] = ErrorMessage(
        String(describing: error), outOfMemory: outOfMemory
    )
    return -1
}

private func context(_ pointer: UnsafeMutableRawPointer) -> Context {
    Unmanaged<Context>.fromOpaque(pointer).takeUnretainedValue()
}

/// Runs `body` with this context to itself. The lock is not recursive, so it is taken here at the
/// edge and nowhere inside: a call that synchronizes on its way already holds it.
private func locked<T>(_ pointer: UnsafeMutableRawPointer, _ body: (Context) -> T) -> T {
    let ctx = context(pointer)
    ctx.lock.lock()
    defer { ctx.lock.unlock() }
    return body(ctx)
}

private func buffer(_ pointer: UnsafeMutableRawPointer) -> MTLBuffer {
    Unmanaged<AnyObject>.fromOpaque(pointer).takeUnretainedValue() as! MTLBuffer
}

private func complete(_ command: MTLCommandBuffer) throws {
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

/// Whether the last failure on this thread was the device having no memory left, which a caller
/// holding memory it could let go of can do something about.
@_cdecl("mmh3_metal_out_of_memory")
func metalOutOfMemory() -> Bool {
    (Thread.current.threadDictionary["mmh3.metal.error"] as? ErrorMessage)?.outOfMemory ?? false
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
        locked(pointer) { ctx in
            do {
                try ctx.synchronize()
                return 0
            } catch {
                return fail(error)
            }
        }
    }
}

/// What one process may fill of this device and what it has filled, which is what a caller asking
/// whether another checkpoint fits has to go on. It takes no context: the numbers belong to the
/// device, and every context here computes on the system default one.
@_cdecl("mmh3_metal_memory_info")
func metalMemoryInfo(_ output: UnsafeMutablePointer<UInt64>) -> Int32 {
    autoreleasepool {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return fail(BridgeError.message("No Metal GPU is available"))
        }

        poolLock.lock()
        let pooledBytes = pooledEverywhere
        poolLock.unlock()
        output[0] = device.recommendedMaxWorkingSetSize
        output[1] = UInt64(max(device.currentAllocatedSize - pooledBytes, 0))
        return 0
    }
}

@_cdecl("mmh3_metal_stats")
func metalStats(_ pointer: UnsafeMutableRawPointer, _ output: UnsafeMutablePointer<UInt64>) {
    locked(pointer) { ctx in
        output[0] = ctx.submissions
        output[1] = ctx.allocations
        output[2] = ctx.reuses
        output[3] = ctx.peakBytes
    }
}

@_cdecl("mmh3_metal_alloc")
func metalAlloc(_ pointer: UnsafeMutableRawPointer, _ bytes: Int, _ data: UnsafeRawPointer?)
    -> UnsafeMutableRawPointer?
{
    autoreleasepool {
        locked(pointer) { ctx in
            do {
                if let error = ctx.executionError {
                    throw error
                }

                // The batches before this one hand back the buffers let go of while they ran, so
                // an allocation this large waits for them, but not for the batch being encoded.
                if ctx.pending != nil, ctx.allocatedSinceSync + bytes > 256 * 1024 * 1024 {
                    try ctx.flush(leaving: 1)
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
                // Pooled buffers count as free, so they go before an allocation takes the device
                // past what one process should fill.
                let share = Int(ctx.device.recommendedMaxWorkingSetSize)
                while !ctx.reusable.isEmpty, ctx.device.currentAllocatedSize + bytes > share {
                    ctx.pooledBytes -= ctx.reusable.removeFirst().length
                }
                guard let allocated = ctx.device.makeBuffer(length: bytes, options: .storageModeShared)
                else {
                    _ = fail(
                        BridgeError.message("Metal buffer allocation failed (\(bytes) bytes)"),
                        outOfMemory: true
                    )
                    return nil
                }

                result = allocated
                ctx.allocations += 1
                ctx.peakBytes = max(ctx.peakBytes, UInt64(ctx.device.currentAllocatedSize))
            }

            ctx.allocatedSinceSync += bytes
            if let data {
                result.contents().copyMemory(from: data, byteCount: bytes)
                // A model's weights go back to the device with the model rather than into the pool.
                result.label = weightsLabel
            } else {
                result.label = nil
            }

            return Unmanaged.passRetained(result as AnyObject).toOpaque()
        }
    }
}

/// A buffer over `bytes` of host memory at `address`, whole pages, which the GPU reads where they
/// lie. Metal calls `release` with `owner` once it lets go of the buffer, after the last command
/// that reads it; so does this call when it cannot make the buffer.
@_cdecl("mmh3_metal_wrap")
func metalWrap(
    _ pointer: UnsafeMutableRawPointer, _ address: UnsafeMutableRawPointer, _ bytes: Int,
    _ release: @escaping @convention(c) (UnsafeRawPointer?) -> Void, _ owner: UnsafeRawPointer?
) -> UnsafeMutableRawPointer? {
    autoreleasepool {
        locked(pointer) { ctx in
            if let error = ctx.executionError {
                release(owner)
                _ = fail(error)
                return nil
            }

            guard let result = ctx.device.makeBuffer(
                bytesNoCopy: address, length: bytes, options: .storageModeShared,
                deallocator: { _, _ in release(owner) }
            ) else {
                release(owner)
                _ = fail(
                    BridgeError.message("Metal could not use \(bytes) bytes of host memory"),
                    outOfMemory: true
                )
                return nil
            }

            // Weights, which go back to the device with the model rather than into the pool.
            result.label = weightsLabel
            ctx.peakBytes = max(ctx.peakBytes, UInt64(ctx.device.currentAllocatedSize))
            return Unmanaged.passRetained(result as AnyObject).toOpaque()
        }
    }
}

// Both drain what they autorelease: a buffer over host memory must go when its last owner lets go,
// since its memory is handed out again only then, and these are called from threads with no pool.
@_cdecl("mmh3_metal_free")
func metalFree(_ ctx: UnsafeMutableRawPointer, _ pointer: UnsafeMutableRawPointer) {
    autoreleasepool {
        let allocation = Unmanaged<AnyObject>.fromOpaque(pointer).takeRetainedValue() as! MTLBuffer
        locked(ctx) { $0.recycle(allocation) }
    }
}

@_cdecl("mmh3_metal_read")
func metalRead(_ pointer: UnsafeMutableRawPointer, _ output: UnsafeMutableRawPointer, _ bytes: Int) {
    autoreleasepool {
        output.copyMemory(from: buffer(pointer).contents(), byteCount: bytes)
    }
}

@_cdecl("mmh3_metal_dispatch")
func metalDispatch(
    _ pointer: UnsafeMutableRawPointer, _ name: UnsafePointer<CChar>,
    _ buffers: UnsafePointer<UnsafeMutableRawPointer>, _ count: Int,
    _ parameters: UnsafeRawPointer, _ bytes: Int, _ threads: Int, _ groupSize: Int, _ groups: Bool
) -> Int32 {
    autoreleasepool {
        locked(pointer) { ctx in
            do {
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

                        guard ctx.fragmentLayout != .unknown else {
                            throw BridgeError.message(
                                "MPP TensorOps lay out their fragments where the kernels do not expect"
                            )
                        }

                        if ctx.tensorLibrary == nil {
                            let options = MTLCompileOptions()
                            options.languageVersion = .version4_0
                            options.mathMode = .safe
                            options.preprocessorMacros = [
                                "FRAGMENT_PAIRS": NSNumber(value: ctx.fragmentLayout == .pairs ? 1 : 0),
                            ]
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
}

@_cdecl("mmh3_metal_matmul")
func metalMatmul(
    _ pointer: UnsafeMutableRawPointer, _ a: UnsafeMutableRawPointer,
    _ b: UnsafeMutableRawPointer, _ c: UnsafeMutableRawPointer, _ m: Int, _ n: Int, _ k: Int,
    _ stride: Int, _ offset: Int, _ halfInputs: Bool
) -> Int32 {
    autoreleasepool {
        locked(pointer) { ctx in
            do {
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
}

@_cdecl("mmh3_metal_supports_tensor_ops")
func metalSupportsTensorOps(_ pointer: UnsafeMutableRawPointer) -> Bool {
    locked(pointer) { $0.fragmentLayout != .unknown }
}
