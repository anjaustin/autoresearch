/*
 * vk_ternary.c -- Minimal Vulkan compute dispatch for ternary matmul
 *
 * Uses the compiled SPIR-V shader (ternary_matmul.spv) to run the
 * ternary matmul on the AMD Vega 7 iGPU.
 *
 * Compile:
 *   gcc -O2 -o softchip/vk_ternary softchip/vk_ternary.c -lvulkan -lm
 *
 * Run:
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json ./softchip/vk_ternary
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

/* -----------------------------------------------------------------------
 * Weight packing (same as CPU kernel)
 * ----------------------------------------------------------------------- */
typedef struct {
    uint32_t *packed;       /* packed as uint32: 16 weights per uint */
    int packed_row_uints;   /* uint32s per row */
    float weight_scale;
} PackedWeights;

static PackedWeights pack_weights_u32(const float *weights, int out_features, int in_features) {
    PackedWeights pw;
    pw.packed_row_uints = (in_features + 15) / 16;

    /* Compute absmean scale */
    double sum_abs = 0.0;
    int total = out_features * in_features;
    for (int i = 0; i < total; i++) sum_abs += fabsf(weights[i]);
    float mean_abs = (float)(sum_abs / total);
    float scale = 1.0f / fmaxf(mean_abs, 1e-5f);
    pw.weight_scale = mean_abs;

    pw.packed = (uint32_t *)calloc(out_features * pw.packed_row_uints, sizeof(uint32_t));

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
 * Push constants struct (must match shader)
 * ----------------------------------------------------------------------- */
typedef struct {
    int32_t in_features;
    int32_t out_features;
    int32_t packed_row_uints;
    float weight_scale;
} PushConstants;

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */
int main(void) {
    const int N = 2560;  /* out_features */
    const int K = 2560;  /* in_features */

    printf("=== Vulkan Ternary Matmul on Vega 7 iGPU ===\n");
    printf("Layer: M=1, K=%d, N=%d\n\n", K, N);

    /* Load Vulkan */
    if (load_vulkan()) return 1;

    /* Create instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "ternary_matmul",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&inst_info, NULL, &instance));
    load_instance_funcs(instance);

    /* Get physical device */
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance, &dev_count, NULL);
    VkPhysicalDevice *phys_devs = malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &dev_count, phys_devs);

    VkPhysicalDevice phys_dev = phys_devs[0];
    VkPhysicalDeviceProperties dev_props;
    vkGetPhysicalDeviceProperties(phys_dev, &dev_props);
    printf("Device: %s\n", dev_props.deviceName);

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

    /* Find compute queue family */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, NULL);
    VkQueueFamilyProperties *qf_props = malloc(qf_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, qf_props);

    uint32_t compute_qf = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_qf = i;
            break;
        }
    }
    if (compute_qf == UINT32_MAX) {
        fprintf(stderr, "No compute queue found\n");
        return 1;
    }

    /* Create logical device */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = compute_qf,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
    };
    VkDevice device;
    VK_CHECK(vkCreateDevice(phys_dev, &dev_info, NULL, &device));

    VkQueue queue;
    vkGetDeviceQueue(device, compute_qf, 0, &queue);

    /* Load SPIR-V */
    size_t spv_size;
    uint32_t *spv_code = load_spirv("softchip/ternary_matmul.spv", &spv_size);

    VkShaderModuleCreateInfo shader_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_size,
        .pCode = spv_code,
    };
    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(device, &shader_info, NULL, &shader));
    free(spv_code);

    /* Generate test data */
    printf("Generating test data...\n");
    srand(42);
    float *weights = aligned_alloc(32, N * K * sizeof(float));
    float *activation = aligned_alloc(32, K * sizeof(float));
    float *output = calloc(N, sizeof(float));

    for (int i = 0; i < N * K; i++)
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < K; i++)
        activation[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    PackedWeights pw = pack_weights_u32(weights, N, K);
    size_t packed_bytes = N * pw.packed_row_uints * sizeof(uint32_t);
    size_t act_bytes = K * sizeof(float);
    size_t out_bytes = N * sizeof(float);

    printf("Packed weights: %zu bytes (%.1f KB)\n", packed_bytes, packed_bytes / 1024.0);
    printf("Weight scale: %.4f\n", pw.weight_scale);

    /* Create buffers */
    VkBuffer buf_packed, buf_act, buf_out;
    VkDeviceMemory mem_packed, mem_act, mem_out;

    VkBufferCreateInfo bi;
    VkMemoryRequirements req;
    VkMemoryAllocateInfo ai;
    VkMemoryPropertyFlags mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

    /* Buffer: packed weights */
    bi.size = packed_bytes;
    VK_CHECK(vkCreateBuffer(device, &bi, NULL, &buf_packed));
    vkGetBufferMemoryRequirements(device, buf_packed, &req);
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(&mem_props, req.memoryTypeBits, mem_flags);
    VK_CHECK(vkAllocateMemory(device, &ai, NULL, &mem_packed));
    VK_CHECK(vkBindBufferMemory(device, buf_packed, mem_packed, 0));

    /* Buffer: activation */
    bi.size = act_bytes;
    VK_CHECK(vkCreateBuffer(device, &bi, NULL, &buf_act));
    vkGetBufferMemoryRequirements(device, buf_act, &req);
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(&mem_props, req.memoryTypeBits, mem_flags);
    VK_CHECK(vkAllocateMemory(device, &ai, NULL, &mem_act));
    VK_CHECK(vkBindBufferMemory(device, buf_act, mem_act, 0));

    /* Buffer: output */
    bi.size = out_bytes;
    VK_CHECK(vkCreateBuffer(device, &bi, NULL, &buf_out));
    vkGetBufferMemoryRequirements(device, buf_out, &req);
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(&mem_props, req.memoryTypeBits, mem_flags);
    VK_CHECK(vkAllocateMemory(device, &ai, NULL, &mem_out));
    VK_CHECK(vkBindBufferMemory(device, buf_out, mem_out, 0));

    /* Upload data */
    void *mapped;
    vkMapMemory(device, mem_packed, 0, packed_bytes, 0, &mapped);
    memcpy(mapped, pw.packed, packed_bytes);
    vkUnmapMemory(device, mem_packed);

    vkMapMemory(device, mem_act, 0, act_bytes, 0, &mapped);
    memcpy(mapped, activation, act_bytes);
    vkUnmapMemory(device, mem_act);

    /* Descriptor set layout */
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
    VkDescriptorSetLayout desc_layout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dsl_info, NULL, &desc_layout));

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
        .pSetLayouts = &desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VkPipelineLayout pipeline_layout;
    VK_CHECK(vkCreatePipelineLayout(device, &pl_info, NULL, &pipeline_layout));

    /* Compute pipeline */
    VkComputePipelineCreateInfo pipe_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = pipeline_layout,
    };
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &pipeline));

    /* Descriptor pool and set */
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3,
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkDescriptorPool desc_pool;
    VK_CHECK(vkCreateDescriptorPool(device, &pool_info, NULL, &desc_pool));

    VkDescriptorSetAllocateInfo desc_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    VK_CHECK(vkAllocateDescriptorSets(device, &desc_alloc, &desc_set));

    /* Update descriptors */
    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = buf_packed, .offset = 0, .range = packed_bytes },
        { .buffer = buf_act, .offset = 0, .range = act_bytes },
        { .buffer = buf_out, .offset = 0, .range = out_bytes },
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
    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

    /* Command pool and buffer */
    VkCommandPoolCreateInfo cmd_pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = compute_qf,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    VkCommandPool cmd_pool;
    VK_CHECK(vkCreateCommandPool(device, &cmd_pool_info, NULL, &cmd_pool));

    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd_buf;
    VK_CHECK(vkAllocateCommandBuffers(device, &cmd_alloc, &cmd_buf));

    /* Fence for synchronization */
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fence_info, NULL, &fence));

    /* Push constants */
    PushConstants pc = {
        .in_features = K,
        .out_features = N,
        .packed_row_uints = pw.packed_row_uints,
        .weight_scale = pw.weight_scale,
    };

    uint32_t workgroups = (N + 255) / 256;

    /* Record command buffer */
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    /* Warmup */
    printf("\nWarming up...\n");
    VK_CHECK(vkBeginCommandBuffer(cmd_buf, &begin_info));
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &desc_set, 0, NULL);
    vkCmdPushConstants(cmd_buf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(cmd_buf, workgroups, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd_buf));

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd_buf,
    };
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(device, 1, &fence));

    /* Benchmark 1: individual submits (includes dispatch overhead) */
    printf("Benchmarking (individual submits)...\n");
    int iters = 200;
    struct timespec t0, t1;

    /* Pre-record command buffer once */
    VK_CHECK(vkResetCommandBuffer(cmd_buf, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd_buf, &begin_info));
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &desc_set, 0, NULL);
    vkCmdPushConstants(cmd_buf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(cmd_buf, workgroups, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd_buf));

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device, 1, &fence));
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double ms_per_call = elapsed / iters * 1000.0;

    /* Benchmark 2: batched dispatches (amortizes submit overhead) */
    printf("Benchmarking (batched dispatches)...\n");
    int batch_dispatches = 7;  /* 7 layers per decoder layer */
    VK_CHECK(vkResetCommandBuffer(cmd_buf, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd_buf, &begin_info));
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &desc_set, 0, NULL);
    vkCmdPushConstants(cmd_buf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    for (int d = 0; d < batch_dispatches; d++) {
        vkCmdDispatch(cmd_buf, workgroups, 1, 1);
    }
    VK_CHECK(vkEndCommandBuffer(cmd_buf));

    int batch_iters = 100;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < batch_iters; i++) {
        VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device, 1, &fence));
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed2 = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double ms_per_batch = elapsed2 / batch_iters * 1000.0;
    double ms_per_dispatch = ms_per_batch / batch_dispatches;

    /* Read back output */
    vkMapMemory(device, mem_out, 0, out_bytes, 0, &mapped);
    memcpy(output, mapped, out_bytes);
    vkUnmapMemory(device, mem_out);

    printf("\n=== Results ===\n");
    printf("Individual submit:  %.3f ms per call\n", ms_per_call);
    printf("Batched (7 dispatch): %.3f ms per batch, %.3f ms per dispatch\n",
           ms_per_batch, ms_per_dispatch);
    printf("Throughput (individual): %.2f GFLOP/s equivalent\n",
           2.0 * N * K / (ms_per_call / 1000.0) / 1e9);
    printf("Throughput (batched):    %.2f GFLOP/s equivalent\n",
           2.0 * N * K / (ms_per_dispatch / 1000.0) / 1e9);
    printf("Output[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
           output[0], output[1], output[2], output[3], output[4]);
    printf("\nComparison:\n");
    printf("  CPU soft-chip (M=1):  1.56 ms\n");
    printf("  GPU individual:       %.3f ms  (%.1fx vs CPU)\n",
           ms_per_call, 1.56 / ms_per_call);
    printf("  GPU batched:          %.3f ms  (%.1fx vs CPU)\n",
           ms_per_dispatch, 1.56 / ms_per_dispatch);
    double submit_overhead = ms_per_call - ms_per_dispatch;
    printf("  Submit overhead:      ~%.3f ms\n", submit_overhead > 0 ? submit_overhead : 0);

    /* Cleanup */
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, cmd_pool, NULL);
    vkDestroyDescriptorPool(device, desc_pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(device, desc_layout, NULL);
    vkDestroyShaderModule(device, shader, NULL);
    vkDestroyBuffer(device, buf_packed, NULL);
    vkDestroyBuffer(device, buf_act, NULL);
    vkDestroyBuffer(device, buf_out, NULL);
    vkFreeMemory(device, mem_packed, NULL);
    vkFreeMemory(device, mem_act, NULL);
    vkFreeMemory(device, mem_out, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    free(phys_devs);
    free(qf_props);
    free(weights);
    free(activation);
    free(output);
    free(pw.packed);
    if (vk_lib) dlclose(vk_lib);

    return 0;
}
