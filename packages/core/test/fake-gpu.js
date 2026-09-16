// A minimal in-memory WebGPU stand-in. It records what the runtime asks for
// and "runs" the model by classifying each packed token on the CPU: digits are
// numbers, symbols are operators, everything else is plain.

export const CLASS = { plain: 0, number: 3, keyword: 4, operator: 8 };

export function defaultClassify(packed, words) {
  const classes = new Uint32Array(packed.length / words);
  for (let i = 0; i < classes.length; i++) {
    const word = packed[i * words];
    const kind = word & 3;
    const first = (word >>> 5) & 127;
    classes[i] = kind === 3 ? CLASS.operator : first >= 48 && first <= 57 ? CLASS.number : CLASS.plain;
  }
  return classes;
}

export function installFakeGpu({
  classify = defaultClassify,
  words = 2,
  adapterLimits = { maxBufferSize: 4_294_967_292, maxStorageBufferBindingSize: 4_294_967_292,
    maxComputeWorkgroupStorageSize: 65_536 },
  compileMessages = [],
  adapter: adapterOverride,
} = {}) {
  const log = {
    adapters: 0, requestedLimits: undefined, code: undefined, entries: [], buffers: [],
    bindGroups: [], dispatches: [], passes: 0, submits: 0, maps: 0, writes: [],
  };
  let bindGroup;
  const device = {
    queue: {
      writeBuffer(buffer, _offset, data) {
        buffer.bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
        log.writes.push(buffer);
      },
      submit() { log.submits += 1; },
    },
    createShaderModule({ code }) {
      log.code = code;
      return { async getCompilationInfo() { return { messages: compileMessages }; } };
    },
    createBindGroupLayout: (descriptor) => descriptor,
    createPipelineLayout: (descriptor) => descriptor,
    async createComputePipelineAsync({ compute }) {
      log.entries.push(compute.entryPoint);
      return { entry: compute.entryPoint };
    },
    createBuffer({ size, usage }) {
      const buffer = {
        size, usage, destroyed: false, bytes: new Uint8Array(size),
        destroy() { this.destroyed = true; },
        async mapAsync() { log.maps += 1; },
        getMappedRange(offset, length) { return this.bytes.buffer.slice(offset, offset + length); },
        unmap() {},
      };
      log.buffers.push(buffer);
      return buffer;
    },
    createBindGroup(descriptor) {
      log.bindGroups.push(descriptor);
      return bindGroup = descriptor;
    },
    createCommandEncoder() {
      return {
        clearBuffer() {},
        beginComputePass() {
          log.passes += 1;
          let pipeline;
          return {
            setPipeline(value) { pipeline = value; },
            setBindGroup() {},
            dispatchWorkgroups(x, y) { log.dispatches.push([pipeline.entry, x, y]); },
            end() {},
          };
        },
        copyBufferToBuffer(_source, _sourceOffset, destination, _offset, size) {
          const tokens = bindGroup.entries[0].resource.buffer;
          const packed = new Uint32Array(tokens.bytes.buffer.slice(0, tokens.bytes.byteLength));
          destination.bytes = new Uint8Array(destination.size);
          new Uint32Array(destination.bytes.buffer, 0, size / 4).set(classify(packed, words).subarray(0, size / 4));
        },
        finish: () => ({}),
      };
    },
  };
  const adapter = adapterOverride ?? {
    limits: adapterLimits,
    async requestDevice(descriptor) {
      log.requestedLimits = descriptor.requiredLimits;
      return device;
    },
  };
  setGpu({ async requestAdapter() { log.adapters += 1; return adapter; } });
  return log;
}

export function setGpu(gpu) {
  Object.defineProperty(globalThis, "navigator", { value: { gpu }, configurable: true, writable: true });
}
