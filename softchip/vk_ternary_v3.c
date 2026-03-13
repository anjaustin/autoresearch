/*
 * vk_ternary_v3.c -- Optimized Vulkan compute dispatch for ternary matmul
 *
 * v3 optimizations:
 *   1. XOR+AND bit trick (no float multiply in inner loop)
 *   2. Fully unrolled vec4 processing (16 weights per uint32, 4 at a time)
 *   3. Specialization constants for LDS right-sizing (2560 vs 6912)
 *   4. Row-major weight layout (same as v2 — A/B testing showed transposed hurts)
 *
 * Benchmarks all 4 BitNet layer shapes with correctness validation.
 * Results: 2x speedup over v2 on 2560x2560 (0.32 ms vs 0.80 ms batched).
 *
 * Compile:
 *   glslangValidator -V softchip/ternary_matmul_v3.comp -o softchip/ternary_matmul_v3.spv
 *   gcc -O2 -o softchip/vk_ternary_v3 softchip/vk_ternary_v3.c -lvulkan -lm -ldl
 *
 * Run:
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json ./softchip/vk_ternary_v3
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

/* -----------------------------------------------------------------------
 * Vulkan function pointers (loaded dynamically)
 * ----------------------------------------------------------------------- */
static void *vk_lib = NULL;

#define VK_FUNC(name) static PFN_##name name = NULL
VK_FUNC(vkGetInstanceProcAddr);
VK_FUNC(vkCreateInstance);
VK_FUNC(vkDestroyInstance);
VK_FUNC(vkEnumeratePhysicalDevices);
VK_FUNC(vkGetPhysicalDeviceProperties);
VK_FUNC(vkGetPhysicalDeviceMemoryProperties);
VK_FUNC(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FUNC(vkCreateDevice);
VK_FUNC(vkDestroyDevice);
VK_FUNC(vkGetDeviceQueue);
VK_FUNC(vkCreateBuffer);
VK_FUNC(vkDestroyBuffer);
VK_FUNC(vkGetBufferMemoryRequirements);
VK_FUNC(vkAllocateMemory);
VK_FUNC(vkFreeMemory);
VK_FUNC(vkBindBufferMemory);
VK_FUNC(vkMapMemory);
VK_FUNC(vkUnmapMemory);
VK_FUNC(vkFlushMappedMemoryRanges);
VK_FUNC(vkInvalidateMappedMemoryRanges);
VK_FUNC(vkCreateShaderModule);
VK_FUNC(vkDestroyShaderModule);
VK_FUNC(vkCreateDescriptorSetLayout);
VK_FUNC(vkDestroyDescriptorSetLayout);
VK_FUNC(vkCreatePipelineLayout);
VK_FUNC(vkDestroyPipelineLayout);
VK_FUNC(vkCreateComputePipelines);
VK_FUNC(vkDestroyPipeline);
VK_FUNC(vkCreateDescriptorPool);
VK_FUNC(vkDestroyDescriptorPool);
VK_FUNC(vkAllocateDescriptorSets);
VK_FUNC(vkUpdateDescriptorSets);
VK_FUNC(vkCreateCommandPool);
VK_FUNC(vkDestroyCommandPool);
VK_FUNC(vkAllocateCommandBuffers);
VK_FUNC(vkBeginCommandBuffer);
VK_FUNC(vkEndCommandBuffer);
VK_FUNC(vkCmdBindPipeline);
VK_FUNC(vkCmdBindDescriptorSets);
VK_FUNC(vkCmdPushConstants);
VK_FUNC(vkCmdDispatch);
VK_FUNC(vkQueueSubmit);
VK_FUNC(vkQueueWaitIdle);
VK_FUNC(vkCreateFence);
VK_FUNC(vkDestroyFence);
VK_FUNC(vkWaitForFences);
VK_FUNC(vkResetFences);
VK_FUNC(vkResetCommandBuffer);
VK_FUNC(vkDeviceWaitIdle);

static int load_vulkan(void) {
    vk_lib = dlopen("libvulkan.so.1", RTLD_NOW);
    if (!vk_lib) { fprintf(stderr, "Failed to load libvulkan.so.1\n"); return -1; }

    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(vk_lib, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) return -1;

    #define LOAD_GLOBAL(fn) fn = (PFN_##fn)vkGetInstanceProcAddr(NULL, #fn)
    LOAD_GLOBAL(vkCreateInstance);
    LOAD_GLOBAL(vkEnumeratePhysicalDevices);
    return 0;
}

static void load_instance_funcs(VkInstance inst) {
    #define LOAD(fn) fn = (PFN_##fn)vkGetInstanceProcAddr(inst, #fn)
    LOAD(vkDestroyInstance);
    LOAD(vkEnumeratePhysicalDevices);
    LOAD(vkGetPhysicalDeviceProperties);
    LOAD(vkGetPhysicalDeviceMemoryProperties);
    LOAD(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD(vkCreateDevice);
    LOAD(vkDestroyDevice);
    LOAD(vkGetDeviceQueue);
    LOAD(vkCreateBuffer);
    LOAD(vkDestroyBuffer);
    LOAD(vkGetBufferMemoryRequirements);
    LOAD(vkAllocateMemory);
    LOAD(vkFreeMemory);
    LOAD(vkBindBufferMemory);
    LOAD(vkMapMemory);
    LOAD(vkUnmapMemory);
    LOAD(vkFlushMappedMemoryRanges);
    LOAD(vkInvalidateMappedMemoryRanges);
    LOAD(vkCreateShaderModule);
    LOAD(vkDestroyShaderModule);
    LOAD(vkCreateDescriptorSetLayout);
    LOAD(vkDestroyDescriptorSetLayout);
    LOAD(vkCreatePipelineLayout);
    LOAD(vkDestroyPipelineLayout);
    LOAD(vkCreateComputePipelines);
    LOAD(vkDestroyPipeline);
    LOAD(vkCreateDescriptorPool);
    LOAD(vkDestroyDescriptorPool);
    LOAD(vkAllocateDescriptorSets);
    LOAD(vkUpdateDescriptorSets);
    LOAD(vkCreateCommandPool);
    LOAD(vkDestroyCommandPool);
    LOAD(vkAllocateCommandBuffers);
    LOAD(vkBeginCommandBuffer);
    LOAD(vkEndCommandBuffer);
    LOAD(vkCmdBindPipeline);
    LOAD(vkCmdBindDescriptorSets);
    LOAD(vkCmdPushConstants);
    LOAD(vkCmdDispatch);
    LOAD(vkQueueSubmit);
    LOAD(vkQueueWaitIdle);
    LOAD(vkCreateFence);
    LOAD(vkDestroyFence);
    LOAD(vkWaitForFences);
    LOAD(vkResetFences);
    LOAD(vkResetCommandBuffer);
    LOAD(vkDeviceWaitIdle);
    #undef LOAD
}

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */
#define VK_CHECK(call) do { VkResult r = (call); if (r != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); } } while(0)

static uint32_t find_memory_type(VkPhysicalDeviceMemoryProperties *props,
                                  uint32_t type_filter, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < props->memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (props->memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    fprintf(stderr, "Failed to find suitable memory type\n");
    exit(1);
}

static uint32_t *load_spirv(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    *out_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *code = (uint32_t *)malloc(*out_size);
    fread(code, 1, *out_size, f);
    fclose(f);
    return code;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* -----------------------------------------------------------------------
 * Weight packing: row-major (for v2 shader, same as original)
 * ----------------------------------------------------------------------- */
typedef struct {
    uint32_t *packed;
    int packed_row_uints;   /* uint32s per row */
    float weight_scale;
} PackedRowMajor;

static PackedRowMajor pack_row_major(const float *weights, int out_features, int in_features) {
    PackedRowMajor pw;
    pw.packed_row_uints = (in_features + 15) / 16;

    double sum_abs = 0.0;
    int total = out_features * in_features;
    for (int i = 0; i < total; i++) sum_abs += fabsf(weights[i]);
    float mean_abs = (float)(sum_abs / total);
    float scale = 1.0f / fmaxf(mean_abs, 1e-5f);
    pw.weight_scale = mean_abs;

    pw.packed = (uint32_t *)calloc((size_t)out_features * pw.packed_row_uints, sizeof(uint32_t));

    for (int row = 0; row < out_features; row++) {
        for (int u = 0; u < pw.packed_row_uints; u++) {
            uint32_t word = 0;
            for (int j = 0; j < 16; j++) {
                int k = u * 16 + j;
                if (k >= in_features) break;
                float wq = roundf(weights[row * in_features + k] * scale);
                if (wq > 1.0f) wq = 1.0f;
                if (wq < -1.0f) wq = -1.0f;
                uint32_t code;
                if (wq > 0.5f)       code = 0x01;
                else if (wq < -0.5f) code = 0x03;
                else                  code = 0x00;
                word |= (code << (j * 2));
            }
            pw.packed[row * pw.packed_row_uints + u] = word;
        }
    }
    return pw;
}

/* -----------------------------------------------------------------------
 * Push constants (must match shader)
 * ----------------------------------------------------------------------- */
typedef struct {
    int32_t in_features;
    int32_t out_features;
    int32_t packed_row_uints;
    float weight_scale;
} PushConstants;

/* -----------------------------------------------------------------------
 * Vulkan context (shared across tests)
 * ----------------------------------------------------------------------- */
typedef struct {
    VkInstance instance;
    VkPhysicalDevice phys_dev;
    VkPhysicalDeviceMemoryProperties mem_props;
    VkDevice device;
    VkQueue queue;
    uint32_t compute_qf;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buf;
    VkFence fence;
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipeline_layout;
    VkDescriptorPool desc_pool;
} VulkanCtx;

static VulkanCtx init_vulkan(void) {
    VulkanCtx ctx;

    if (load_vulkan()) exit(1);

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "ternary_matmul_v3",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    VK_CHECK(vkCreateInstance(&inst_info, NULL, &ctx.instance));
    load_instance_funcs(ctx.instance);

    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &dev_count, NULL);
    VkPhysicalDevice *phys_devs = malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx.instance, &dev_count, phys_devs);
    ctx.phys_dev = phys_devs[0];
    free(phys_devs);

    VkPhysicalDeviceProperties dev_props;
    vkGetPhysicalDeviceProperties(ctx.phys_dev, &dev_props);
    printf("Device: %s\n\n", dev_props.deviceName);

    vkGetPhysicalDeviceMemoryProperties(ctx.phys_dev, &ctx.mem_props);

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys_dev, &qf_count, NULL);
    VkQueueFamilyProperties *qf_props = malloc(qf_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys_dev, &qf_count, qf_props);

    ctx.compute_qf = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx.compute_qf = i;
            break;
        }
    }
    free(qf_props);
    if (ctx.compute_qf == UINT32_MAX) {
        fprintf(stderr, "No compute queue found\n");
        exit(1);
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx.compute_qf,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
    };
    VK_CHECK(vkCreateDevice(ctx.phys_dev, &dev_info, NULL, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.compute_qf, 0, &ctx.queue);

    /* Descriptor set layout (3 storage buffers) */
    VkDescriptorSetLayoutBinding bindings[3] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };
    VkDescriptorSetLayoutCreateInfo dsl_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dsl_info, NULL, &ctx.desc_layout));

    /* Push constant range */
    VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };

    /* Pipeline layout */
    VkPipelineLayoutCreateInfo pl_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx.desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VK_CHECK(vkCreatePipelineLayout(ctx.device, &pl_info, NULL, &ctx.pipeline_layout));

    /* Command pool */
    VkCommandPoolCreateInfo cmd_pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx.compute_qf,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    VK_CHECK(vkCreateCommandPool(ctx.device, &cmd_pool_info, NULL, &ctx.cmd_pool));

    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx.cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cmd_alloc, &ctx.cmd_buf));

    /* Fence */
    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(vkCreateFence(ctx.device, &fence_info, NULL, &ctx.fence));

    /* Descriptor pool (enough for multiple descriptor sets) */
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 30,  /* plenty */
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 10,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &pool_info, NULL, &ctx.desc_pool));

    return ctx;
}

static void destroy_vulkan(VulkanCtx *ctx) {
    vkDestroyFence(ctx->device, ctx->fence, NULL);
    vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
    vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
    vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
    vkDestroyDevice(ctx->device, NULL);
    vkDestroyInstance(ctx->instance, NULL);
    if (vk_lib) dlclose(vk_lib);
}

/* -----------------------------------------------------------------------
 * Create a compute pipeline from SPIR-V, with optional specialization
 * ----------------------------------------------------------------------- */
static VkPipeline create_pipeline(VulkanCtx *ctx, const char *spv_path,
                                   VkPipelineLayout layout,
                                   uint32_t spec_in_features) {
    size_t spv_size;
    uint32_t *spv_code = load_spirv(spv_path, &spv_size);

    VkShaderModuleCreateInfo shader_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_size,
        .pCode = spv_code,
    };
    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(ctx->device, &shader_info, NULL, &shader));
    free(spv_code);

    /* Specialization constant: SPEC_IN_FEATURES (id=0) */
    VkSpecializationMapEntry spec_entry = {
        .constantID = 0,
        .offset = 0,
        .size = sizeof(uint32_t),
    };
    VkSpecializationInfo spec_info = {
        .mapEntryCount = 1,
        .pMapEntries = &spec_entry,
        .dataSize = sizeof(uint32_t),
        .pData = &spec_in_features,
    };

    VkComputePipelineCreateInfo pipe_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
            .pSpecializationInfo = (spec_in_features > 0) ? &spec_info : NULL,
        },
        .layout = layout,
    };
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &pipeline));
    vkDestroyShaderModule(ctx->device, shader, NULL);

    return pipeline;
}

/* -----------------------------------------------------------------------
 * Buffer helpers
 * ----------------------------------------------------------------------- */
typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
} GpuBuffer;

static GpuBuffer create_buffer(VulkanCtx *ctx, size_t size) {
    GpuBuffer gb;
    gb.size = size;

    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(ctx->device, &bi, NULL, &gb.buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx->device, gb.buffer, &req);

    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = find_memory_type(&ctx->mem_props, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    VK_CHECK(vkAllocateMemory(ctx->device, &ai, NULL, &gb.memory));
    VK_CHECK(vkBindBufferMemory(ctx->device, gb.buffer, gb.memory, 0));

    return gb;
}

static void upload_buffer(VulkanCtx *ctx, GpuBuffer *gb, const void *data, size_t size) {
    void *mapped;
    vkMapMemory(ctx->device, gb->memory, 0, size, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory(ctx->device, gb->memory);
}

static void download_buffer(VulkanCtx *ctx, GpuBuffer *gb, void *data, size_t size) {
    void *mapped;
    vkMapMemory(ctx->device, gb->memory, 0, size, 0, &mapped);
    memcpy(data, mapped, size);
    vkUnmapMemory(ctx->device, gb->memory);
}

static void destroy_buffer(VulkanCtx *ctx, GpuBuffer *gb) {
    vkDestroyBuffer(ctx->device, gb->buffer, NULL);
    vkFreeMemory(ctx->device, gb->memory, NULL);
}

/* -----------------------------------------------------------------------
 * Allocate and bind descriptor set
 * ----------------------------------------------------------------------- */
static VkDescriptorSet create_desc_set(VulkanCtx *ctx, GpuBuffer *packed,
                                         GpuBuffer *act, GpuBuffer *out) {
    VkDescriptorSetAllocateInfo desc_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &ctx->desc_layout,
    };
    VkDescriptorSet desc_set;
    VK_CHECK(vkAllocateDescriptorSets(ctx->device, &desc_alloc, &desc_set));

    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = packed->buffer, .offset = 0, .range = packed->size },
        { .buffer = act->buffer, .offset = 0, .range = act->size },
        { .buffer = out->buffer, .offset = 0, .range = out->size },
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = desc_set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buf_infos[i],
        };
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
    return desc_set;
}

/* -----------------------------------------------------------------------
 * Run a dispatch and time it
 * ----------------------------------------------------------------------- */
static double run_benchmark(VulkanCtx *ctx, VkPipeline pipeline,
                             VkDescriptorSet desc_set, PushConstants *pc,
                             uint32_t workgroups, int iters) {
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    /* Pre-record */
    VK_CHECK(vkResetCommandBuffer(ctx->cmd_buf, 0));
    VK_CHECK(vkBeginCommandBuffer(ctx->cmd_buf, &begin_info));
    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layout, 0, 1, &desc_set, 0, NULL);
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), pc);
    vkCmdDispatch(ctx->cmd_buf, workgroups, 1, 1);
    VK_CHECK(vkEndCommandBuffer(ctx->cmd_buf));

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd_buf,
    };

    /* Warmup */
    VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submit, ctx->fence));
    VK_CHECK(vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(ctx->device, 1, &ctx->fence));

    /* Timed */
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submit, ctx->fence));
        VK_CHECK(vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(ctx->device, 1, &ctx->fence));
    }
    double elapsed = now_sec() - t0;
    return elapsed / iters * 1000.0;  /* ms per call */
}

/* -----------------------------------------------------------------------
 * Run a batched dispatch (N dispatches in one command buffer)
 * ----------------------------------------------------------------------- */
static double run_benchmark_batched(VulkanCtx *ctx, VkPipeline pipeline,
                                     VkDescriptorSet desc_set, PushConstants *pc,
                                     uint32_t workgroups, int batch_dispatches, int iters) {
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    VK_CHECK(vkResetCommandBuffer(ctx->cmd_buf, 0));
    VK_CHECK(vkBeginCommandBuffer(ctx->cmd_buf, &begin_info));
    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layout, 0, 1, &desc_set, 0, NULL);
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), pc);
    for (int d = 0; d < batch_dispatches; d++) {
        vkCmdDispatch(ctx->cmd_buf, workgroups, 1, 1);
    }
    VK_CHECK(vkEndCommandBuffer(ctx->cmd_buf));

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd_buf,
    };

    /* Warmup */
    VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submit, ctx->fence));
    VK_CHECK(vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(ctx->device, 1, &ctx->fence));

    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submit, ctx->fence));
        VK_CHECK(vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(ctx->device, 1, &ctx->fence));
    }
    double elapsed = now_sec() - t0;
    return elapsed / iters * 1000.0 / batch_dispatches;  /* ms per dispatch */
}

/* -----------------------------------------------------------------------
 * Test one layer shape
 * ----------------------------------------------------------------------- */
typedef struct {
    const char *name;
    int out_features;
    int in_features;
} LayerShape;

static void test_layer(VulkanCtx *ctx, VkPipeline pipe_v3, LayerShape *shape) {
    int N = shape->out_features;
    int K = shape->in_features;

    printf("--- %s (%dx%d) ---\n", shape->name, N, K);

    /* Generate test data */
    srand(42);
    float *weights = aligned_alloc(32, (size_t)N * K * sizeof(float));
    float *activation = aligned_alloc(32, K * sizeof(float));
    float *gpu_output = calloc(N, sizeof(float));
    float *cpu_output = calloc(N, sizeof(float));

    for (int i = 0; i < N * K; i++)
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < K; i++)
        activation[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    /* Pack weights row-major (for v3c shader) */
    PackedRowMajor rm = pack_row_major(weights, N, K);

    /* CPU reference using row-major weights */
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            int u = k / 16, j = k % 16;
            uint32_t word = rm.packed[n * rm.packed_row_uints + u];
            uint32_t code = (word >> (j * 2)) & 3u;
            float nz = (float)(code & 1u);
            float sign = 1.0f - 2.0f * (float)(code >> 1u);
            acc += activation[k] * nz * sign;
        }
        cpu_output[n] = acc * rm.weight_scale;
    }

    /* Upload to GPU */
    size_t rm_bytes = (size_t)N * rm.packed_row_uints * sizeof(uint32_t);
    size_t act_bytes = K * sizeof(float);
    size_t out_bytes = N * sizeof(float);

    GpuBuffer buf_packed = create_buffer(ctx, rm_bytes);
    GpuBuffer buf_act = create_buffer(ctx, act_bytes);
    GpuBuffer buf_out = create_buffer(ctx, out_bytes);

    upload_buffer(ctx, &buf_packed, rm.packed, rm_bytes);
    upload_buffer(ctx, &buf_act, activation, act_bytes);

    /* Descriptor set */
    VkDescriptorSet desc_set = create_desc_set(ctx, &buf_packed, &buf_act, &buf_out);

    /* Push constants */
    PushConstants pc = {
        .in_features = K,
        .out_features = N,
        .packed_row_uints = rm.packed_row_uints,
        .weight_scale = rm.weight_scale,
    };
    uint32_t workgroups = (N + 255) / 256;

    /* Run once to get output for validation */
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    VK_CHECK(vkResetCommandBuffer(ctx->cmd_buf, 0));
    VK_CHECK(vkBeginCommandBuffer(ctx->cmd_buf, &begin_info));
    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_v3);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layout, 0, 1, &desc_set, 0, NULL);
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), &pc);
    vkCmdDispatch(ctx->cmd_buf, workgroups, 1, 1);
    VK_CHECK(vkEndCommandBuffer(ctx->cmd_buf));

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd_buf,
    };
    VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submit, ctx->fence));
    VK_CHECK(vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(ctx->device, 1, &ctx->fence));

    download_buffer(ctx, &buf_out, gpu_output, out_bytes);

    /* Validate GPU vs CPU */
    double max_diff = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = fabs(gpu_output[i] - cpu_output[i]);
        if (diff > max_diff) max_diff = diff;
        sum_sq_diff += diff * diff;
        sum_sq_ref += (double)cpu_output[i] * cpu_output[i];
    }
    double nrmse = sqrt(sum_sq_diff / (sum_sq_ref + 1e-10));
    printf("  GPU vs CPU:  max_diff=%.6f  NRMSE=%.2e  %s\n",
           max_diff, nrmse, (nrmse < 1e-4) ? "PASS" : "FAIL");

    /* Benchmark: individual submits */
    int iters = (N <= 1024) ? 500 : 200;
    double ms_individual = run_benchmark(ctx, pipe_v3, desc_set, &pc, workgroups, iters);

    /* Benchmark: batched (7 dispatches) */
    double ms_batched = run_benchmark_batched(ctx, pipe_v3, desc_set, &pc,
                                              workgroups, 7, iters / 2);

    double gflops_ind = 2.0 * N * K / (ms_individual / 1000.0) / 1e9;
    double gflops_bat = 2.0 * N * K / (ms_batched / 1000.0) / 1e9;

    printf("  Individual submit:  %.3f ms  (%.1f GFLOP/s eq)\n", ms_individual, gflops_ind);
    printf("  Batched (7 disp):   %.3f ms  (%.1f GFLOP/s eq)\n", ms_batched, gflops_bat);
    printf("  Submit overhead:    ~%.3f ms\n",
           ms_individual - ms_batched > 0 ? ms_individual - ms_batched : 0);
    printf("\n");

    /* Cleanup */
    destroy_buffer(ctx, &buf_packed);
    destroy_buffer(ctx, &buf_act);
    destroy_buffer(ctx, &buf_out);
    free(weights);
    free(activation);
    free(gpu_output);
    free(cpu_output);
    free(rm.packed);
}

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */
int main(void) {
    printf("=== Vulkan Ternary Matmul v3: BitTrick + Unrolled vec4 + LDS Sizing ===\n\n");

    VulkanCtx ctx = init_vulkan();

    /* Create v3 pipeline with specialization constant for 2560 (q/k/v/o_proj, gate/up_proj) */
    VkPipeline pipe_v3_2560 = create_pipeline(&ctx, "softchip/ternary_matmul_v3.spv",
                                                ctx.pipeline_layout, 2560);
    /* Create v3 pipeline with specialization constant for 6912 (down_proj) */
    VkPipeline pipe_v3_6912 = create_pipeline(&ctx, "softchip/ternary_matmul_v3.spv",
                                                ctx.pipeline_layout, 6912);

    printf("Pipelines created: 2560-variant (60%% occupancy) and 6912-variant (20%% occupancy)\n\n");

    /* Test all BitNet layer shapes */
    LayerShape shapes[] = {
        { "q_proj / o_proj", 2560, 2560 },
        { "k_proj / v_proj",  640, 2560 },
        { "gate_proj / up_proj", 6912, 2560 },
        { "down_proj", 2560, 6912 },
    };

    for (int i = 0; i < 4; i++) {
        VkPipeline pipe = (shapes[i].in_features <= 2560) ? pipe_v3_2560 : pipe_v3_6912;
        test_layer(&ctx, pipe, &shapes[i]);
    }

    /* Summary comparison with v2 baselines */
    printf("=== Comparison with v2 baseline (from previous run) ===\n");
    printf("  v2 (2560x2560, batched):  0.61 ms/dispatch\n");
    printf("  v2 (2560x2560, individual): 1.04 ms/dispatch\n");
    printf("  CPU soft-chip v3 (M=1):   1.56 ms\n");
    printf("\n  Target for v3: 0.15-0.25 ms (2560x2560 batched)\n");
    printf("  Hard gate: v3 must be at least 2x faster than v2 (< 0.30 ms)\n");

    /* Cleanup */
    vkDestroyPipeline(ctx.device, pipe_v3_2560, NULL);
    vkDestroyPipeline(ctx.device, pipe_v3_6912, NULL);
    destroy_vulkan(&ctx);

    return 0;
}
