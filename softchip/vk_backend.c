/*
 * vk_backend.c -- Vulkan compute backend for ternary matmul (shared library)
 *
 * Provides a C API for Python/ctypes:
 *   vk_init()           -- Initialize Vulkan device, pipelines
 *   vk_alloc_layer()    -- Upload packed weights for one layer
 *   vk_dispatch()       -- Run ternary matmul on GPU for one layer
 *   vk_dispatch_batch() -- Run multiple independent matmuls in one submit
 *   vk_shutdown()       -- Free all resources
 *
 * Compile:
 *   glslangValidator -V softchip/ternary_matmul_v3.comp -o softchip/ternary_matmul_v3.spv
 *   gcc -O2 -shared -fPIC -o softchip/libvk_ternary.so softchip/vk_backend.c -lvulkan -lm -ldl
 *
 * Weight format: row-major packed uint32, same as CPU kernel and v3 shader.
 * 2 bits per weight, 16 per uint32. packed_row_uints = ceil(in_features/16).
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

/* -----------------------------------------------------------------------
 * Vulkan function pointers
 * ----------------------------------------------------------------------- */
static void *vk_lib = NULL;

#define VK_FUNC(name) static PFN_##name name = NULL
VK_FUNC(vkGetInstanceProcAddr);
VK_FUNC(vkCreateInstance); VK_FUNC(vkDestroyInstance);
VK_FUNC(vkEnumeratePhysicalDevices); VK_FUNC(vkGetPhysicalDeviceProperties);
VK_FUNC(vkGetPhysicalDeviceMemoryProperties);
VK_FUNC(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FUNC(vkCreateDevice); VK_FUNC(vkDestroyDevice); VK_FUNC(vkGetDeviceQueue);
VK_FUNC(vkCreateBuffer); VK_FUNC(vkDestroyBuffer);
VK_FUNC(vkGetBufferMemoryRequirements);
VK_FUNC(vkAllocateMemory); VK_FUNC(vkFreeMemory);
VK_FUNC(vkBindBufferMemory); VK_FUNC(vkMapMemory); VK_FUNC(vkUnmapMemory);
VK_FUNC(vkCreateShaderModule); VK_FUNC(vkDestroyShaderModule);
VK_FUNC(vkCreateDescriptorSetLayout); VK_FUNC(vkDestroyDescriptorSetLayout);
VK_FUNC(vkCreatePipelineLayout); VK_FUNC(vkDestroyPipelineLayout);
VK_FUNC(vkCreateComputePipelines); VK_FUNC(vkDestroyPipeline);
VK_FUNC(vkCreateDescriptorPool); VK_FUNC(vkDestroyDescriptorPool);
VK_FUNC(vkAllocateDescriptorSets); VK_FUNC(vkUpdateDescriptorSets);
VK_FUNC(vkCreateCommandPool); VK_FUNC(vkDestroyCommandPool);
VK_FUNC(vkAllocateCommandBuffers);
VK_FUNC(vkBeginCommandBuffer); VK_FUNC(vkEndCommandBuffer);
VK_FUNC(vkCmdBindPipeline); VK_FUNC(vkCmdBindDescriptorSets);
VK_FUNC(vkCmdPushConstants); VK_FUNC(vkCmdDispatch);
VK_FUNC(vkQueueSubmit); VK_FUNC(vkQueueWaitIdle);
VK_FUNC(vkCreateFence); VK_FUNC(vkDestroyFence);
VK_FUNC(vkWaitForFences); VK_FUNC(vkResetFences);
VK_FUNC(vkResetCommandBuffer); VK_FUNC(vkDeviceWaitIdle);
VK_FUNC(vkResetDescriptorPool);

/* -----------------------------------------------------------------------
 * Constants
 * ----------------------------------------------------------------------- */
#define MAX_LAYERS 256
#define MAX_BATCH 8

#define VK_CHECK(call) do { VkResult r = (call); if (r != VK_SUCCESS) { \
    fprintf(stderr, "VK error %d at %s:%d\n", r, __FILE__, __LINE__); return -1; } } while(0)

#define VK_CHECK_VOID(call) do { VkResult r = (call); if (r != VK_SUCCESS) { \
    fprintf(stderr, "VK error %d at %s:%d\n", r, __FILE__, __LINE__); return; } } while(0)

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
 * Per-layer GPU state
 * ----------------------------------------------------------------------- */
typedef struct {
    VkBuffer weight_buf;
    VkDeviceMemory weight_mem;
    int out_features;
    int in_features;
    int packed_row_uints;
    float weight_scale;
    size_t weight_bytes;
    int active;
} LayerState;

/* -----------------------------------------------------------------------
 * Global Vulkan state
 * ----------------------------------------------------------------------- */
static struct {
    int initialized;

    VkInstance instance;
    VkPhysicalDevice phys_dev;
    VkPhysicalDeviceMemoryProperties mem_props;
    VkDevice device;
    VkQueue queue;
    uint32_t compute_qf;

    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipe_2560;       /* LDS sized for in_features <= 2560 */
    VkPipeline pipe_6912;       /* LDS sized for in_features <= 6912 */

    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buf;
    VkFence fence;

    VkDescriptorPool desc_pool;       /* For pre-allocated per-layer descriptor sets */
    VkDescriptorPool batch_desc_pool; /* For temporary batch descriptor sets */

    /* Activation and output staging buffers (shared, resized as needed) */
    VkBuffer act_buf;
    VkDeviceMemory act_mem;
    size_t act_capacity;
    void *act_mapped;           /* Persistently mapped activation pointer */

    VkBuffer out_buf;
    VkDeviceMemory out_mem;
    size_t out_capacity;
    void *out_mapped;           /* Persistently mapped output pointer */

    /* For batch: multiple output buffers */
    VkBuffer batch_out_bufs[MAX_BATCH];
    VkDeviceMemory batch_out_mems[MAX_BATCH];
    size_t batch_out_caps[MAX_BATCH];
    void *batch_out_mapped[MAX_BATCH];  /* Persistently mapped */

    /* Pre-allocated descriptor sets for all layers (avoid alloc/reset per dispatch) */
    VkDescriptorSet layer_desc_sets[MAX_LAYERS];
    int desc_sets_allocated;

    LayerState layers[MAX_LAYERS];
    int num_layers;

    char device_name[256];
} G = {0};

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */
static uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < G.mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (G.mem_props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    return UINT32_MAX;
}

static int create_buffer(VkBuffer *buf, VkDeviceMemory *mem, size_t size) {
    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(G.device, &bi, NULL, buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.device, *buf, &req);

    VkMemoryPropertyFlags mf = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t mt = find_memory_type(req.memoryTypeBits, mf);
    if (mt == UINT32_MAX) return -1;

    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mt,
    };
    VK_CHECK(vkAllocateMemory(G.device, &ai, NULL, mem));
    VK_CHECK(vkBindBufferMemory(G.device, *buf, *mem, 0));
    return 0;
}

static void destroy_buffer(VkBuffer *buf, VkDeviceMemory *mem) {
    if (*buf) { vkDestroyBuffer(G.device, *buf, NULL); *buf = VK_NULL_HANDLE; }
    if (*mem) { vkFreeMemory(G.device, *mem, NULL); *mem = VK_NULL_HANDLE; }
}

static int ensure_buffer(VkBuffer *buf, VkDeviceMemory *mem, size_t *cap, size_t needed) {
    if (*cap >= needed) return 0;
    destroy_buffer(buf, mem);
    int rc = create_buffer(buf, mem, needed);
    if (rc == 0) *cap = needed;
    return rc;
}

/* Ensure buffer with persistent mapping */
static int ensure_buffer_mapped(VkBuffer *buf, VkDeviceMemory *mem, size_t *cap,
                                 void **mapped, size_t needed) {
    if (*cap >= needed) return 0;
    if (*mapped) { vkUnmapMemory(G.device, *mem); *mapped = NULL; }
    destroy_buffer(buf, mem);
    int rc = create_buffer(buf, mem, needed);
    if (rc != 0) return rc;
    *cap = needed;
    VkResult r = vkMapMemory(G.device, *mem, 0, needed, 0, mapped);
    if (r != VK_SUCCESS) { *mapped = NULL; return -1; }
    return 0;
}

static uint32_t *load_spirv(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *out_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *code = (uint32_t *)malloc(*out_size);
    if (fread(code, 1, *out_size, f) != *out_size) { free(code); fclose(f); return NULL; }
    fclose(f);
    return code;
}

static VkPipeline create_pipeline_spec(const char *spv_path, uint32_t spec_val) {
    size_t sz;
    uint32_t *code = load_spirv(spv_path, &sz);
    if (!code) return VK_NULL_HANDLE;

    VkShaderModuleCreateInfo smi = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sz, .pCode = code,
    };
    VkShaderModule sm;
    VkResult r = vkCreateShaderModule(G.device, &smi, NULL, &sm);
    free(code);
    if (r != VK_SUCCESS) return VK_NULL_HANDLE;

    VkSpecializationMapEntry se = { 0, 0, sizeof(uint32_t) };
    VkSpecializationInfo si = { 1, &se, sizeof(uint32_t), &spec_val };

    VkComputePipelineCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm, .pName = "main",
            .pSpecializationInfo = &si,
        },
        .layout = G.pipeline_layout,
    };
    VkPipeline pipe;
    r = vkCreateComputePipelines(G.device, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe);
    vkDestroyShaderModule(G.device, sm, NULL);
    return (r == VK_SUCCESS) ? pipe : VK_NULL_HANDLE;
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

/**
 * Initialize Vulkan backend. Returns 0 on success, -1 on failure.
 * spv_path: path to compiled ternary_matmul_v3.spv shader.
 */
int vk_init(const char *spv_path) {
    if (G.initialized) return 0;

    /* Load Vulkan library */
    vk_lib = dlopen("libvulkan.so.1", RTLD_NOW);
    if (!vk_lib) { fprintf(stderr, "vk_backend: cannot load libvulkan.so.1\n"); return -1; }

    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(vk_lib, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) return -1;

    #define LG(fn) fn = (PFN_##fn)vkGetInstanceProcAddr(NULL, #fn)
    LG(vkCreateInstance); LG(vkEnumeratePhysicalDevices);
    #undef LG

    /* Create instance */
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "softchip_vk",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VK_CHECK(vkCreateInstance(&ici, NULL, &G.instance));

    /* Load instance functions */
    #define L(fn) fn = (PFN_##fn)vkGetInstanceProcAddr(G.instance, #fn)
    L(vkDestroyInstance); L(vkEnumeratePhysicalDevices);
    L(vkGetPhysicalDeviceProperties); L(vkGetPhysicalDeviceMemoryProperties);
    L(vkGetPhysicalDeviceQueueFamilyProperties);
    L(vkCreateDevice); L(vkDestroyDevice); L(vkGetDeviceQueue);
    L(vkCreateBuffer); L(vkDestroyBuffer); L(vkGetBufferMemoryRequirements);
    L(vkAllocateMemory); L(vkFreeMemory); L(vkBindBufferMemory);
    L(vkMapMemory); L(vkUnmapMemory);
    L(vkCreateShaderModule); L(vkDestroyShaderModule);
    L(vkCreateDescriptorSetLayout); L(vkDestroyDescriptorSetLayout);
    L(vkCreatePipelineLayout); L(vkDestroyPipelineLayout);
    L(vkCreateComputePipelines); L(vkDestroyPipeline);
    L(vkCreateDescriptorPool); L(vkDestroyDescriptorPool);
    L(vkAllocateDescriptorSets); L(vkUpdateDescriptorSets);
    L(vkCreateCommandPool); L(vkDestroyCommandPool);
    L(vkAllocateCommandBuffers);
    L(vkBeginCommandBuffer); L(vkEndCommandBuffer);
    L(vkCmdBindPipeline); L(vkCmdBindDescriptorSets);
    L(vkCmdPushConstants); L(vkCmdDispatch);
    L(vkQueueSubmit); L(vkQueueWaitIdle);
    L(vkCreateFence); L(vkDestroyFence);
    L(vkWaitForFences); L(vkResetFences);
    L(vkResetCommandBuffer); L(vkDeviceWaitIdle);
    L(vkResetDescriptorPool);
    #undef L

    /* Physical device */
    uint32_t dc = 0;
    vkEnumeratePhysicalDevices(G.instance, &dc, NULL);
    if (dc == 0) return -1;
    VkPhysicalDevice *pds = malloc(dc * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(G.instance, &dc, pds);
    G.phys_dev = pds[0];
    free(pds);

    VkPhysicalDeviceProperties dp;
    vkGetPhysicalDeviceProperties(G.phys_dev, &dp);
    strncpy(G.device_name, dp.deviceName, sizeof(G.device_name) - 1);
    vkGetPhysicalDeviceMemoryProperties(G.phys_dev, &G.mem_props);

    /* Compute queue */
    uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys_dev, &qfc, NULL);
    VkQueueFamilyProperties *qfp = malloc(qfc * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys_dev, &qfc, qfp);
    G.compute_qf = UINT32_MAX;
    for (uint32_t i = 0; i < qfc; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { G.compute_qf = i; break; }
    }
    free(qfp);
    if (G.compute_qf == UINT32_MAX) return -1;

    /* Logical device */
    float qp = 1.0f;
    VkDeviceQueueCreateInfo qi = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = G.compute_qf, .queueCount = 1, .pQueuePriorities = &qp,
    };
    VkDeviceCreateInfo di = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
    };
    VK_CHECK(vkCreateDevice(G.phys_dev, &di, NULL, &G.device));
    vkGetDeviceQueue(G.device, G.compute_qf, 0, &G.queue);

    /* Descriptor set layout (3 storage buffers) */
    VkDescriptorSetLayoutBinding bnd[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dli = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bnd,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(G.device, &dli, NULL, &G.desc_layout));

    /* Pipeline layout */
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants) };
    VkPipelineLayoutCreateInfo pli = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &G.desc_layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
    };
    VK_CHECK(vkCreatePipelineLayout(G.device, &pli, NULL, &G.pipeline_layout));

    /* Create two pipelines: LDS sized for 2560 and 6912 */
    G.pipe_2560 = create_pipeline_spec(spv_path, 2560);
    G.pipe_6912 = create_pipeline_spec(spv_path, 6912);
    if (!G.pipe_2560 || !G.pipe_6912) {
        fprintf(stderr, "vk_backend: failed to create pipelines from %s\n", spv_path);
        return -1;
    }

    /* Command pool + buffer */
    VkCommandPoolCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = G.compute_qf,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    VK_CHECK(vkCreateCommandPool(G.device, &cpi, NULL, &G.cmd_pool));

    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = G.cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(G.device, &cai, &G.cmd_buf));

    /* Fence */
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(vkCreateFence(G.device, &fi, NULL, &G.fence));

    /* Descriptor pool for pre-allocated per-layer sets */
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_LAYERS * 3 };
    VkDescriptorPoolCreateInfo dpi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = MAX_LAYERS,
        .poolSizeCount = 1, .pPoolSizes = &dps,
    };
    VK_CHECK(vkCreateDescriptorPool(G.device, &dpi, NULL, &G.desc_pool));

    /* Separate descriptor pool for batch dispatch (resettable) */
    VkDescriptorPoolSize dps2 = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_BATCH * 3 };
    VkDescriptorPoolCreateInfo dpi2 = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = MAX_BATCH,
        .poolSizeCount = 1, .pPoolSizes = &dps2,
    };
    VK_CHECK(vkCreateDescriptorPool(G.device, &dpi2, NULL, &G.batch_desc_pool));

    G.initialized = 1;
    G.num_layers = 0;
    return 0;
}

/**
 * Get device name. Returns NULL if not initialized.
 */
const char *vk_device_name(void) {
    return G.initialized ? G.device_name : NULL;
}

/**
 * Upload packed weights for one layer. Returns layer_id (0-based), or -1 on error.
 *
 * packed_weights: row-major packed uint32 array (same format as CPU kernel)
 * out_features, in_features: layer dimensions
 * weight_scale: absmean scale factor
 */
int vk_alloc_layer(const uint32_t *packed_weights, int out_features, int in_features,
                    float weight_scale) {
    if (!G.initialized) return -1;
    if (G.num_layers >= MAX_LAYERS) return -1;

    int id = G.num_layers;
    LayerState *ls = &G.layers[id];

    ls->out_features = out_features;
    ls->in_features = in_features;
    ls->packed_row_uints = (in_features + 15) / 16;
    ls->weight_scale = weight_scale;
    ls->weight_bytes = (size_t)out_features * ls->packed_row_uints * sizeof(uint32_t);

    if (create_buffer(&ls->weight_buf, &ls->weight_mem, ls->weight_bytes) != 0)
        return -1;

    /* Upload */
    void *mapped;
    VkResult r = vkMapMemory(G.device, ls->weight_mem, 0, ls->weight_bytes, 0, &mapped);
    if (r != VK_SUCCESS) return -1;
    memcpy(mapped, packed_weights, ls->weight_bytes);
    vkUnmapMemory(G.device, ls->weight_mem);

    ls->active = 1;
    G.num_layers++;
    return id;
}

/**
 * Ensure layer has a pre-allocated descriptor set.
 * Called once per layer; descriptor set is reused across dispatches
 * by updating it when activation/output buffer addresses change.
 */
static int ensure_layer_desc_set(int layer_id) {
    if (G.desc_sets_allocated) return 0;  /* Already allocated for all layers */
    return 0;  /* Will be allocated in finalize_layers() */
}

/**
 * Track last bound buffers to avoid redundant descriptor set updates.
 */
static VkBuffer last_act_buf = VK_NULL_HANDLE;
static VkBuffer last_out_buf = VK_NULL_HANDLE;

static void update_desc_set_io(int layer_id);  /* Forward declaration */

/**
 * Pre-allocate descriptor sets for all layers and set weight bindings.
 * Call after all layers have been uploaded.
 */
int vk_finalize_layers(void) {
    if (!G.initialized || G.num_layers == 0) return -1;
    if (G.desc_sets_allocated) return 0;

    /* Find max dimensions to pre-allocate act/out buffers */
    size_t max_act = 0, max_out = 0;
    for (int i = 0; i < G.num_layers; i++) {
        size_t ab = G.layers[i].in_features * sizeof(float);
        size_t ob = G.layers[i].out_features * sizeof(float);
        if (ab > max_act) max_act = ab;
        if (ob > max_out) max_out = ob;
    }

    /* Pre-allocate activation and output buffers (avoids resize during dispatch) */
    if (ensure_buffer(&G.act_buf, &G.act_mem, &G.act_capacity, max_act) != 0) return -1;
    if (ensure_buffer(&G.out_buf, &G.out_mem, &G.out_capacity, max_out) != 0) return -1;

    /* Allocate descriptor sets for all layers */
    VkDescriptorSetLayout layouts[MAX_LAYERS];
    for (int i = 0; i < G.num_layers; i++) layouts[i] = G.desc_layout;

    VkDescriptorSetAllocateInfo dai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = G.desc_pool,
        .descriptorSetCount = G.num_layers,
        .pSetLayouts = layouts,
    };
    VK_CHECK(vkAllocateDescriptorSets(G.device, &dai, G.layer_desc_sets));

    /* Bind weight + activation + output buffers for all layers */
    for (int i = 0; i < G.num_layers; i++) {
        LayerState *ls = &G.layers[i];
        VkDescriptorBufferInfo bi[3] = {
            { ls->weight_buf, 0, ls->weight_bytes },
            { G.act_buf, 0, VK_WHOLE_SIZE },
            { G.out_buf, 0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet wr[3];
        for (int j = 0; j < 3; j++) {
            wr[j] = (VkWriteDescriptorSet){
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = G.layer_desc_sets[i], .dstBinding = j, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bi[j],
            };
        }
        vkUpdateDescriptorSets(G.device, 3, wr, 0, NULL);
    }

    /* Mark buffers as bound so update_desc_set_io won't re-update */
    last_act_buf = G.act_buf;
    last_out_buf = G.out_buf;

    G.desc_sets_allocated = 1;
    return 0;
}

/**
 * Update activation and output buffer bindings for a layer's descriptor set.
 * Uses VK_WHOLE_SIZE so all layers can share the same descriptor set bindings
 * regardless of their actual data size (shader uses push constants for dimensions).
 * Only updates when the underlying Vulkan buffer handle changes (after resize).
 */
static void update_desc_set_io(int layer_id) {
    /* Only update if buffer handles changed (i.e., after a resize) */
    if (G.act_buf == last_act_buf && G.out_buf == last_out_buf) return;

    last_act_buf = G.act_buf;
    last_out_buf = G.out_buf;

    /* Update ALL layer descriptor sets since the buffer handles changed */
    for (int i = 0; i < G.num_layers; i++) {
        if (!G.layer_desc_sets[i]) continue;
        VkDescriptorBufferInfo bi[2] = {
            { G.act_buf, 0, VK_WHOLE_SIZE },
            { G.out_buf, 0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet wr[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = G.layer_desc_sets[i], .dstBinding = 1, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bi[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = G.layer_desc_sets[i], .dstBinding = 2, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bi[1],
            },
        };
        vkUpdateDescriptorSets(G.device, 2, wr, 0, NULL);
    }
}

/**
 * Run a single ternary matmul on the GPU.
 *
 * layer_id: from vk_alloc_layer()
 * activation: float array of size in_features (input)
 * output: float array of size out_features (output, written by GPU)
 *
 * Returns 0 on success, -1 on error.
 */
int vk_dispatch(int layer_id, const float *activation, float *output) {
    if (!G.initialized || layer_id < 0 || layer_id >= G.num_layers) return -1;
    LayerState *ls = &G.layers[layer_id];
    if (!ls->active) return -1;

    size_t act_bytes = ls->in_features * sizeof(float);
    size_t out_bytes = ls->out_features * sizeof(float);

    /* Ensure staging buffers are large enough */
    if (ensure_buffer(&G.act_buf, &G.act_mem, &G.act_capacity, act_bytes) != 0) return -1;
    if (ensure_buffer(&G.out_buf, &G.out_mem, &G.out_capacity, out_bytes) != 0) return -1;

    /* Upload activation */
    void *mapped;
    vkMapMemory(G.device, G.act_mem, 0, act_bytes, 0, &mapped);
    memcpy(mapped, activation, act_bytes);
    vkUnmapMemory(G.device, G.act_mem);

    /* Allocate descriptor set on first use per layer (lazy finalization) */
    if (!G.layer_desc_sets[layer_id]) {
        VkDescriptorSetAllocateInfo dai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &G.desc_layout,
        };
        VK_CHECK(vkAllocateDescriptorSets(G.device, &dai, &G.layer_desc_sets[layer_id]));

        /* Bind weight buffer (never changes) */
        VkDescriptorBufferInfo wbi = { ls->weight_buf, 0, ls->weight_bytes };
        VkWriteDescriptorSet wwr = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = G.layer_desc_sets[layer_id], .dstBinding = 0, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &wbi,
        };
        vkUpdateDescriptorSets(G.device, 1, &wwr, 0, NULL);
    }

    /* Always update activation and output bindings (exact sizes per layer) */
    {
        VkDescriptorBufferInfo bi[2] = {
            { G.act_buf, 0, act_bytes },
            { G.out_buf, 0, out_bytes },
        };
        VkWriteDescriptorSet wr[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = G.layer_desc_sets[layer_id], .dstBinding = 1, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bi[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = G.layer_desc_sets[layer_id], .dstBinding = 2, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bi[1],
            },
        };
        vkUpdateDescriptorSets(G.device, 2, wr, 0, NULL);
    }

    /* Select pipeline based on in_features */
    VkPipeline pipe = (ls->in_features <= 2560) ? G.pipe_2560 : G.pipe_6912;

    PushConstants pc = {
        .in_features = ls->in_features,
        .out_features = ls->out_features,
        .packed_row_uints = ls->packed_row_uints,
        .weight_scale = ls->weight_scale,
    };
    uint32_t wgs = (ls->out_features + 255) / 256;

    /* Record and submit */
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkResetCommandBuffer(G.cmd_buf, 0));
    VK_CHECK(vkBeginCommandBuffer(G.cmd_buf, &begin));
    vkCmdBindPipeline(G.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(G.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            G.pipeline_layout, 0, 1, &G.layer_desc_sets[layer_id], 0, NULL);
    vkCmdPushConstants(G.cmd_buf, G.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd_buf, wgs, 1, 1);
    VK_CHECK(vkEndCommandBuffer(G.cmd_buf));

    VkSubmitInfo sub = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd_buf,
    };
    VK_CHECK(vkQueueSubmit(G.queue, 1, &sub, G.fence));
    VK_CHECK(vkWaitForFences(G.device, 1, &G.fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(G.device, 1, &G.fence));

    /* Read back output */
    vkMapMemory(G.device, G.out_mem, 0, out_bytes, 0, &mapped);
    memcpy(output, mapped, out_bytes);
    vkUnmapMemory(G.device, G.out_mem);

    /* Descriptor sets are reused across dispatches (no reset needed) */

    return 0;
}

/**
 * Run multiple independent matmuls in a single command buffer submission.
 * All layers share the same activation input but write to separate outputs.
 *
 * layer_ids: array of layer IDs
 * count: number of layers (max MAX_BATCH)
 * activation: shared input float array (size = max in_features across layers)
 * outputs: array of float* output pointers (one per layer)
 *
 * Returns 0 on success, -1 on error.
 */
int vk_dispatch_batch(const int *layer_ids, int count, const float *activation,
                       float **outputs) {
    if (!G.initialized || count <= 0 || count > MAX_BATCH) return -1;

    /* Find max in_features for activation buffer sizing */
    size_t max_act_bytes = 0;
    for (int i = 0; i < count; i++) {
        int id = layer_ids[i];
        if (id < 0 || id >= G.num_layers || !G.layers[id].active) return -1;
        size_t ab = G.layers[id].in_features * sizeof(float);
        if (ab > max_act_bytes) max_act_bytes = ab;
    }

    /* Ensure activation buffer */
    if (ensure_buffer(&G.act_buf, &G.act_mem, &G.act_capacity, max_act_bytes) != 0)
        return -1;

    /* Upload activation */
    void *mapped;
    vkMapMemory(G.device, G.act_mem, 0, max_act_bytes, 0, &mapped);
    memcpy(mapped, activation, max_act_bytes);
    vkUnmapMemory(G.device, G.act_mem);

    /* Ensure per-batch output buffers */
    for (int i = 0; i < count; i++) {
        size_t ob = G.layers[layer_ids[i]].out_features * sizeof(float);
        if (ensure_buffer(&G.batch_out_bufs[i], &G.batch_out_mems[i],
                          &G.batch_out_caps[i], ob) != 0)
            return -1;
    }

    /* Allocate descriptor sets (one per layer in batch) */
    VkDescriptorSetLayout layouts[MAX_BATCH];
    for (int i = 0; i < count; i++) layouts[i] = G.desc_layout;

    VkDescriptorSetAllocateInfo dai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = G.batch_desc_pool,
        .descriptorSetCount = count,
        .pSetLayouts = layouts,
    };
    VkDescriptorSet ds[MAX_BATCH];
    VK_CHECK(vkAllocateDescriptorSets(G.device, &dai, ds));

    /* Update descriptor sets */
    for (int i = 0; i < count; i++) {
        LayerState *ls = &G.layers[layer_ids[i]];
        size_t out_bytes = ls->out_features * sizeof(float);
        size_t act_bytes = ls->in_features * sizeof(float);

        VkDescriptorBufferInfo bi[3] = {
            { ls->weight_buf, 0, ls->weight_bytes },
            { G.act_buf, 0, act_bytes },
            { G.batch_out_bufs[i], 0, out_bytes },
        };
        VkWriteDescriptorSet wr[3];
        for (int j = 0; j < 3; j++) {
            wr[j] = (VkWriteDescriptorSet){
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = ds[i], .dstBinding = j, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bi[j],
            };
        }
        vkUpdateDescriptorSets(G.device, 3, wr, 0, NULL);
    }

    /* Record command buffer with all dispatches */
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkResetCommandBuffer(G.cmd_buf, 0));
    VK_CHECK(vkBeginCommandBuffer(G.cmd_buf, &begin));

    for (int i = 0; i < count; i++) {
        LayerState *ls = &G.layers[layer_ids[i]];
        VkPipeline pipe = (ls->in_features <= 2560) ? G.pipe_2560 : G.pipe_6912;

        PushConstants pc = {
            .in_features = ls->in_features,
            .out_features = ls->out_features,
            .packed_row_uints = ls->packed_row_uints,
            .weight_scale = ls->weight_scale,
        };
        uint32_t wgs = (ls->out_features + 255) / 256;

        vkCmdBindPipeline(G.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(G.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                G.pipeline_layout, 0, 1, &ds[i], 0, NULL);
        vkCmdPushConstants(G.cmd_buf, G.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(G.cmd_buf, wgs, 1, 1);
    }

    VK_CHECK(vkEndCommandBuffer(G.cmd_buf));

    /* Submit */
    VkSubmitInfo sub = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd_buf,
    };
    VK_CHECK(vkQueueSubmit(G.queue, 1, &sub, G.fence));
    VK_CHECK(vkWaitForFences(G.device, 1, &G.fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(G.device, 1, &G.fence));

    /* Read back outputs */
    for (int i = 0; i < count; i++) {
        size_t ob = G.layers[layer_ids[i]].out_features * sizeof(float);
        vkMapMemory(G.device, G.batch_out_mems[i], 0, ob, 0, &mapped);
        memcpy(outputs[i], mapped, ob);
        vkUnmapMemory(G.device, G.batch_out_mems[i]);
    }

    /* Reset batch descriptor pool (doesn't affect pre-allocated layer sets) */
    vkResetDescriptorPool(G.device, G.batch_desc_pool, 0);

    return 0;
}

/**
 * Shut down Vulkan backend and free all resources.
 */
void vk_shutdown(void) {
    if (!G.initialized) return;

    vkDeviceWaitIdle(G.device);

    /* Free layer buffers */
    for (int i = 0; i < G.num_layers; i++) {
        if (G.layers[i].active) {
            destroy_buffer(&G.layers[i].weight_buf, &G.layers[i].weight_mem);
        }
    }

    /* Unmap and free staging buffers */
    if (G.act_mapped) { vkUnmapMemory(G.device, G.act_mem); G.act_mapped = NULL; }
    if (G.out_mapped) { vkUnmapMemory(G.device, G.out_mem); G.out_mapped = NULL; }
    destroy_buffer(&G.act_buf, &G.act_mem);
    destroy_buffer(&G.out_buf, &G.out_mem);
    for (int i = 0; i < MAX_BATCH; i++) {
        if (G.batch_out_mapped[i]) { vkUnmapMemory(G.device, G.batch_out_mems[i]); G.batch_out_mapped[i] = NULL; }
        destroy_buffer(&G.batch_out_bufs[i], &G.batch_out_mems[i]);
    }

    vkDestroyFence(G.device, G.fence, NULL);
    vkDestroyCommandPool(G.device, G.cmd_pool, NULL);
    vkDestroyDescriptorPool(G.device, G.desc_pool, NULL);
    vkDestroyDescriptorPool(G.device, G.batch_desc_pool, NULL);
    vkDestroyPipeline(G.device, G.pipe_2560, NULL);
    vkDestroyPipeline(G.device, G.pipe_6912, NULL);
    vkDestroyPipelineLayout(G.device, G.pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(G.device, G.desc_layout, NULL);
    vkDestroyDevice(G.device, NULL);
    vkDestroyInstance(G.instance, NULL);

    if (vk_lib) { dlclose(vk_lib); vk_lib = NULL; }

    memset(&G, 0, sizeof(G));
}
