use ash::extensions;
use ash::util;
use ash::vk;
use ash::Device;
use ash::Entry;
use ash::Instance;
use cgmath::Matrix4;
use context::VkContext;
use debug::{populate_debug_messenger_create_info, ValidationInfo};
use queries::QueueFamilyIndices;
use std::mem::align_of;
use std::{ffi::CString, os::raw::c_void, ptr};
use vertex::Vertex;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
mod context;
mod debug;
mod helpers;
mod queries;
pub mod vertex;

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct CameraUBO {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

impl CameraUBO {
    fn get_descriptor_set_layout_binding() -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: ptr::null(),
        }
    }
}

const VERTICES_DATA: [Vertex; 4] = [
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.0],
        color: [1.0, 1.0, 1.0],
    },
];

const INDICES_DATA: [u32; 6] = [0, 1, 2, 2, 3, 0];

const VERTEX_SHADER: &str = "
#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}";

const FRAGMENT_SHADER: &str = "
#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}";

pub const VALIDATION: ValidationInfo = ValidationInfo {
    is_enabled: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};

const DEVICE_EXTENSIONS: queries::DeviceExtension = queries::DeviceExtension {
    names: ["VK_KHR_swapchain"],
};

pub struct SurfaceData {
    pub surface: vk::SurfaceKHR,
    pub surface_loader: extensions::khr::Surface,
    pub width: u32,
    pub height: u32,
}

struct SwapChainData {
    swapchain_loader: extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
}

struct SyncObjects {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    inflight_fences: Vec<vk::Fence>,
}

pub struct Renderer {
    context: VkContext,
    debug_utils_loader: extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    queue_family: QueueFamilyIndices,
    gfx_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_loader: extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_imageviews: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    gfx_pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_pool_transient: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    is_framebuffer_resized: bool,
    vertex_buffer: vk::Buffer,
    vertex_buffer_mem: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_mem: vk::DeviceMemory,
    ind_buf_len: u32,
    ubos: Vec<vk::Buffer>,
    ubo_mems: Vec<vk::DeviceMemory>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Renderer {
        // init vulkan stuff
        unsafe {
            let entry = Entry::load().unwrap();
            if VALIDATION.is_enabled && !queries::check_validation_layer_support(&entry) {
                panic!("Validation layer requested but it's not available");
            }
            let instance = Renderer::create_instance(&entry);
            let (debug_utils_loader, debug_messenger) =
                Renderer::setup_debug_utils(&entry, &instance);
            let surface_data = Renderer::create_surface(&entry, &instance, window);
            let gpu = Renderer::pick_suitable_physical_device(&instance, &surface_data);
            let (device, indices) = Renderer::create_logical_device(&instance, gpu, &surface_data);
            let context = VkContext::new(entry, instance, surface_data, gpu, device);
            let gfx_queue = context
                .device
                .get_device_queue(indices.graphics_family.unwrap(), 0);
            let present_queue = context
                .device
                .get_device_queue(indices.present_family.unwrap(), 0);
            let swapchain_data = Renderer::create_swap_chain(&context, &indices);
            let swapchain_imageviews = Renderer::create_image_views(
                &context.device,
                swapchain_data.swapchain_format,
                &swapchain_data.swapchain_images,
            );
            let render_pass =
                Renderer::create_render_pass(&context.device, swapchain_data.swapchain_format);
            let descriptor_set_bindings = [CameraUBO::get_descriptor_set_layout_binding()];
            let descriptor_set_layouts = [Renderer::create_descriptor_set_layout(
                &context.device,
                &descriptor_set_bindings,
            )];
            let (gfx_pipeline, pipeline_layout) = Renderer::create_graphics_pipeline(
                &context.device,
                render_pass,
                swapchain_data.swapchain_extent,
                &descriptor_set_layouts,
            );

            let framebuffers = Renderer::create_framebuffers(
                &context.device,
                render_pass,
                &swapchain_imageviews,
                &swapchain_data.swapchain_extent,
            );

            let command_pool = Renderer::create_command_pool(
                &context.device,
                &indices,
                vk::CommandPoolCreateFlags::empty(),
            );

            let command_pool_transient = Renderer::create_command_pool(
                &context.device,
                &indices,
                vk::CommandPoolCreateFlags::TRANSIENT,
            );

            let (vertex_buffer, vertex_buffer_mem) = Renderer::create_vertex_buffer(
                &context,
                command_pool_transient,
                gfx_queue,
                &VERTICES_DATA,
            );
            let (index_buffer, index_buffer_mem) = Renderer::create_index_buffer(
                &context,
                command_pool_transient,
                gfx_queue,
                &INDICES_DATA,
            );

            let (ubos, ubo_mems) = Renderer::create_uniform_buffer::<CameraUBO>(
                &context,
                swapchain_data.swapchain_images.len(),
            );
            let descriptor_pool = Renderer::create_descriptor_pool(
                &context.device,
                swapchain_data.swapchain_images.len() as u32,
            );
            let descriptor_sets = Renderer::create_descriptor_sets(
                &context.device,
                descriptor_pool,
                descriptor_set_layouts[0],
                &ubos,
            );

            let command_buffers = Renderer::create_command_buffers(
                &context.device,
                command_pool,
                gfx_pipeline,
                render_pass,
                &framebuffers,
                swapchain_data.swapchain_extent,
                vertex_buffer,
                index_buffer,
                INDICES_DATA.len() as u32,
                pipeline_layout,
                &descriptor_sets,
            );

            let sync_object = Renderer::create_sync_objects(&context.device);
            Renderer {
                context,
                debug_utils_loader,
                debug_messenger,
                queue_family: indices,
                gfx_queue,
                present_queue,
                swapchain: swapchain_data.swapchain,
                swapchain_loader: swapchain_data.swapchain_loader,
                swapchain_images: swapchain_data.swapchain_images,
                swapchain_format: swapchain_data.swapchain_format,
                swapchain_extent: swapchain_data.swapchain_extent,
                swapchain_imageviews,
                render_pass,
                pipeline_layout,
                gfx_pipeline,
                framebuffers,
                command_pool,
                command_pool_transient,
                command_buffers,
                image_available_semaphores: sync_object.image_available_semaphores,
                render_finished_semaphores: sync_object.render_finished_semaphores,
                in_flight_fences: sync_object.inflight_fences,
                current_frame: 0,
                is_framebuffer_resized: false,
                vertex_buffer,
                vertex_buffer_mem,
                index_buffer,
                index_buffer_mem,
                ind_buf_len: INDICES_DATA.len() as u32,
                ubos,
                ubo_mems,
                descriptor_set_layout: descriptor_set_layouts[0],
                descriptor_pool,
                descriptor_sets,
            }
        }
    }

    pub fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title("Sapfire Renderer")
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(event_loop)
            .expect("Failed to create window.")
    }

    pub fn main_loop(mut self, window: winit::window::Window, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } => match (virtual_keycode, state) {
                        (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                            *control_flow = ControlFlow::Exit
                        }
                        _ => {}
                    },
                },
                WindowEvent::Resized(size) => {
                    self.recreate_swapchain(size);
                }
                _ => {}
            },
            Event::RedrawEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_windowid) => self.draw_frame(),
            Event::LoopDestroyed => unsafe {
                self.context
                    .device
                    .device_wait_idle()
                    .expect("Failed to wait for device to become idle");
            },

            _ => (),
        })
    }

    fn draw_frame(&mut self) {
        let wait_fences = [self.in_flight_fences[self.current_frame]];
        unsafe {
            self.context
                .device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for fences");
        };
        let (image_index, _) = unsafe {
            let result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );
            match result {
                Ok(image_index) => image_index,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.recreate_swapchain_default();
                        return;
                    }
                    _ => {
                        panic!("Failed to acquire swapchain image");
                    }
                },
            }
        };
        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[image_index as usize],
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        }];
        unsafe {
            self.context
                .device
                .reset_fences(&wait_fences)
                .expect("Failed to reset wait fences");
            self.context
                .device
                .queue_submit(
                    self.gfx_queue,
                    &submit_infos,
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to submit queue");
        };
        let swapchains = [self.swapchain];
        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
        };
        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        };
        let is_resized = match result {
            Ok(_) => self.is_framebuffer_resized,
            Err(vk_res) => match vk_res {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present"),
            },
        };
        if is_resized {
            self.is_framebuffer_resized = false;
            self.recreate_swapchain_default();
        }
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn recreate_swapchain_default(&mut self) {
        self.recreate_swapchain(winit::dpi::PhysicalSize::<u32> {
            width: WINDOW_WIDTH,
            height: WINDOW_HEIGHT,
        });
    }
    fn recreate_swapchain(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        let surface_data = SurfaceData {
            surface: self.context.surface_data.surface,
            surface_loader: self.context.surface_data.surface_loader.clone(),
            width: size.width,
            height: size.height,
        };
        self.context.surface_data = surface_data;
        unsafe {
            self.context
                .device
                .device_wait_idle()
                .expect("Failed to wait for the device to become idle");
        }
        self.drop_swapchain();
        let swapchain_data = Renderer::create_swap_chain(&self.context, &self.queue_family);
        self.swapchain = swapchain_data.swapchain;
        self.swapchain_extent = swapchain_data.swapchain_extent;
        self.swapchain_format = swapchain_data.swapchain_format;
        self.swapchain_loader = swapchain_data.swapchain_loader;
        self.swapchain_images = swapchain_data.swapchain_images;
        self.swapchain_imageviews = Renderer::create_image_views(
            &self.context.device,
            self.swapchain_format,
            &self.swapchain_images,
        );
        self.render_pass =
            Renderer::create_render_pass(&self.context.device, self.swapchain_format);
        (self.gfx_pipeline, self.pipeline_layout) = Renderer::create_graphics_pipeline(
            &self.context.device,
            self.render_pass,
            swapchain_data.swapchain_extent,
            &[self.descriptor_set_layout],
        );
        self.framebuffers = Renderer::create_framebuffers(
            &self.context.device,
            self.render_pass,
            &self.swapchain_imageviews,
            &self.swapchain_extent,
        );
        self.command_buffers = Renderer::create_command_buffers(
            &self.context.device,
            self.command_pool,
            self.gfx_pipeline,
            self.render_pass,
            &self.framebuffers,
            self.swapchain_extent,
            self.vertex_buffer,
            self.index_buffer,
            self.ind_buf_len,
            self.pipeline_layout,
            &self.descriptor_sets,
        )
    }

    fn create_instance(entry: &Entry) -> ash::Instance {
        let application_name = CString::new("Sapfire").unwrap();
        let engine_name = CString::new("Sapfire Engine").unwrap();
        let app_info = vk::ApplicationInfo {
            api_version: vk::API_VERSION_1_0,
            engine_version: 0,
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: application_name.as_ptr(),
            p_engine_name: engine_name.as_ptr(),
            application_version: 0,
        };
        let debug_utils_create_info = populate_debug_messenger_create_info();
        let extension_names = required_instance_extension_names();
        let required_validation_names_raw: Vec<CString> = VALIDATION
            .required_validation_layers
            .iter()
            .map(|name| CString::new(*name).unwrap())
            .collect();
        let enabled_layer_names: Vec<*const i8> = required_validation_names_raw
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        #[cfg(target_os = "macos")]
        {
            let create_info = vk::InstanceCreateInfo {
                enabled_extension_count: extension_names.len() as u32,
                pp_enabled_extension_names: extension_names.as_ptr(),
                enabled_layer_count: if VALIDATION.is_enabled {
                    enabled_layer_names.len() as u32
                } else {
                    0
                },
                pp_enabled_layer_names: if VALIDATION.is_enabled {
                    enabled_layer_names.as_ptr()
                } else {
                    ptr::null()
                },
                p_next: if VALIDATION.is_enabled {
                    &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                        as *const c_void
                } else {
                    ptr::null()
                },
                s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                flags: vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
                p_application_info: &app_info,
            };
            let instance: Instance = unsafe {
                entry
                    .create_instance(&create_info, None)
                    .expect("Failed to create Vulkan instance!")
            };
            instance
        }
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        {
            let create_info = vk::InstanceCreateInfo {
                enabled_extension_count: extension_names.len() as u32,
                pp_enabled_extension_names: extension_names.as_ptr(),
                enabled_layer_count: if VALIDATION.is_enabled {
                    enabled_layer_names.len() as u32
                } else {
                    0
                },
                pp_enabled_layer_names: if VALIDATION.is_enabled {
                    enabled_layer_names.as_ptr()
                } else {
                    ptr::null()
                },
                p_next: if VALIDATION.is_enabled {
                    &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                        as *const c_void
                } else {
                    ptr::null()
                },
                s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                flags: vk::InstanceCreateFlags::empty(),
                p_application_info: &app_info,
            };
            let instance: Instance = unsafe {
                entry
                    .create_instance(&create_info, None)
                    .expect("Failed to create Vulkan instance!")
            };
            instance
        }
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface_data: &SurfaceData,
    ) -> (Device, QueueFamilyIndices) {
        let indices = queries::find_queue_family(instance, physical_device, surface_data);
        let mut unique_families = std::collections::HashSet::new();
        unique_families.insert(indices.graphics_family);
        unique_families.insert(indices.present_family);
        let q_prios = [1.0_f32];
        let extension_names = required_device_extension_names();
        let mut q_create_infos = vec![];
        for &q_fam in unique_families.iter() {
            let queue_create_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                flags: vk::DeviceQueueCreateFlags::empty(),
                p_next: ptr::null(),
                p_queue_priorities: q_prios.as_ptr(),
                queue_count: q_prios.len() as u32,
                queue_family_index: q_fam.unwrap(),
            };
            q_create_infos.push(queue_create_info);
        }

        let gpu_device_features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };
        let required_validation_names_raw: Vec<CString> = VALIDATION
            .required_validation_layers
            .iter()
            .map(|name| CString::new(*name).unwrap())
            .collect();
        let enabled_layer_names: Vec<*const i8> = required_validation_names_raw
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            enabled_layer_count: enabled_layer_names.len() as u32,
            pp_enabled_layer_names: if VALIDATION.is_enabled {
                enabled_layer_names.as_ptr()
            } else {
                ptr::null()
            },
            pp_enabled_extension_names: extension_names.as_ptr(),
            enabled_extension_count: extension_names.len() as u32,
            p_enabled_features: &gpu_device_features,
            p_queue_create_infos: q_create_infos.as_ptr(),
            queue_create_info_count: q_create_infos.len() as u32,
            p_next: ptr::null(),
            flags: vk::DeviceCreateFlags::empty(),
        };
        let device: Device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };
        (device, indices)
    }

    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &winit::window::Window,
    ) -> SurfaceData {
        let surface = unsafe { create_surface(entry, instance, window).unwrap() };
        let surface_loader = extensions::khr::Surface::new(entry, instance);
        SurfaceData {
            surface,
            surface_loader,
            width: window.inner_size().width,
            height: window.inner_size().height,
        }
    }

    fn create_swap_chain(context: &VkContext, queue_family: &QueueFamilyIndices) -> SwapChainData {
        let swapchain_support =
            queries::query_swapchain_support(context.physical_device, &context.surface_data);
        let surface_format = Renderer::pick_swapchain_format(&swapchain_support.formats);
        let present_mode = Renderer::pick_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = Renderer::pick_swapchain_extent(&swapchain_support.capabilities);
        let (image_sharing_mode, queue_family_index_count, queue_family_indexes) =
            if queue_family.graphics_family != queue_family.present_family {
                (
                    vk::SharingMode::CONCURRENT,
                    2,
                    vec![
                        queue_family.graphics_family.unwrap(),
                        queue_family.present_family.unwrap(),
                    ],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, vec![])
            };
        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            surface: context.surface_data.surface,
            min_image_count: swapchain_support.capabilities.min_image_count + 1,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            queue_family_index_count,
            p_queue_family_indices: queue_family_indexes.as_ptr(),
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
        };
        let swapchain_loader = extensions::khr::Swapchain::new(&context.instance, &context.device);
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create swapchain!")
        };
        let swapchain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images")
        };
        SwapChainData {
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_format: surface_format.format,
            swapchain_extent: extent,
        }
    }

    fn create_image_views(
        device: &Device,
        swapchain_format: vk::Format,
        swapchain_images: &Vec<vk::Image>,
    ) -> Vec<vk::ImageView> {
        let mut image_views = vec![];
        for &image in swapchain_images {
            let imageview_create_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                image,
                format: swapchain_format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: vk::ComponentMapping {
                    a: vk::ComponentSwizzle::IDENTITY,
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            let image_view = unsafe {
                device
                    .create_image_view(&imageview_create_info, None)
                    .expect("Failed to create image view")
            };
            image_views.push(image_view);
        }
        image_views
    }

    fn create_render_pass(device: &Device, swapchain_format: vk::Format) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            format: swapchain_format,
            flags: vk::AttachmentDescriptionFlags::empty(),
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        };
        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let subpass_descr = vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_resolve_attachments: ptr::null(),
            p_depth_stencil_attachment: ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        };
        let render_pass_attachments = [color_attachment];
        let subpass_dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        }];
        let renderpass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: render_pass_attachments.len() as u32,
            p_attachments: render_pass_attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass_descr,
            dependency_count: subpass_dependencies.len() as u32,
            p_dependencies: subpass_dependencies.as_ptr(),
        };
        unsafe {
            device
                .create_render_pass(&renderpass_create_info, None)
                .expect("Failed to create render pass")
        }
    }

    fn create_graphics_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        swapchain_extent: vk::Extent2D,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> (vk::Pipeline, ash::vk::PipelineLayout) {
        let compiler = shaderc::Compiler::new().unwrap(); // TODO: move shader compilation to its own function/module
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.add_macro_definition("EP", Some("main"));
        let vert_code = compiler
            .compile_into_spirv(
                VERTEX_SHADER,
                shaderc::ShaderKind::Vertex,
                "shader.glsl",
                "main",
                Some(&options),
            )
            .unwrap()
            .as_binary_u8()
            .to_vec();
        let frag_code = compiler
            .compile_into_spirv(
                FRAGMENT_SHADER,
                shaderc::ShaderKind::Fragment,
                "shader.glsl",
                "main",
                Some(&options),
            )
            .unwrap()
            .as_binary_u8()
            .to_vec();
        let vert_shader_module = Renderer::create_shader_module(device, vert_code);
        let frag_shader_module = Renderer::create_shader_module(device, frag_code);
        let main_function_name = CString::new("main").unwrap();
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
            },
        ];
        // let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        // let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
        //     s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        //     p_next: ptr::null(),
        //     flags: vk::PipelineDynamicStateCreateFlags::empty(),
        //     dynamic_state_count: dynamic_states.len() as u32,
        //     p_dynamic_states: dynamic_states.as_ptr(),
        // };
        let vertex_binding_descr = [Vertex::binding_description()];
        let vertext_attrib_descr = Vertex::attribute_description();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            // since I set the vertex data directly in the shader
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_binding_description_count: vertex_binding_descr.len() as u32,
            p_vertex_binding_descriptions: vertex_binding_descr.as_ptr(),
            vertex_attribute_description_count: vertext_attrib_descr.len() as u32,
            p_vertex_attribute_descriptions: vertext_attrib_descr.as_ptr(),
        };
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
        };
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        }];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
        };
        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: vk::FALSE, // will clamp depth value instead of discarding on fragments beyond near and far planes. Useful in shadowmapping.
            rasterizer_discard_enable: vk::FALSE, // TRUE will not allow geometry to pass through rasterizer stage
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };
        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            // should be turned off for now
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 1.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
        };
        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState {
            // for alpha blending = bad performance, so turned off
            color_write_mask: vk::ColorComponentFlags::RGBA,
            blend_enable: vk::FALSE,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
        }];
        let color_blend_info = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: color_blend_attachment.len() as u32,
            p_attachments: color_blend_attachment.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: descriptor_set_layouts.len() as u32,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
        };
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create a pipeline layout")
        };
        let pipeline_infos = [vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly_info,
            p_tessellation_state: ptr::null(),
            p_viewport_state: &viewport_state_info,
            p_rasterization_state: &rasterizer_info,
            p_multisample_state: &multisampling,
            p_depth_stencil_state: ptr::null(),
            p_color_blend_state: &color_blend_info,
            p_dynamic_state: ptr::null(),
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
        }];
        let gfx_pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
                .expect("Failed to create graphics pipeline")
        };
        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }
        (gfx_pipelines[0], pipeline_layout)
    }

    fn create_shader_module(device: &Device, code: Vec<u8>) -> ash::vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            flags: vk::ShaderModuleCreateFlags::empty(),
            p_next: ptr::null(),
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: code.len(),
            p_code: code.as_ptr() as *const u32,
        };
        unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create shader module")
        }
    }

    fn create_framebuffers(
        device: &Device,
        render_pass: vk::RenderPass,
        swapchain_imageviews: &Vec<vk::ImageView>,
        swapchain_extent: &vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        swapchain_imageviews
            .iter()
            .map(|&view| {
                let attachments = [view];
                let create_info = vk::FramebufferCreateInfo {
                    s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: vk::FramebufferCreateFlags::empty(),
                    render_pass,
                    attachment_count: attachments.len() as u32,
                    p_attachments: attachments.as_ptr(),
                    width: swapchain_extent.width,
                    height: swapchain_extent.height,
                    layers: 1,
                };
                let framebuffer = unsafe {
                    device
                        .create_framebuffer(&create_info, None)
                        .expect("Failed to create framebuffer")
                };
                framebuffer
            })
            .collect()
    }

    fn create_vertex_buffer(
        context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        data: &[Vertex],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        Renderer::create_device_local_buffer_with_data::<u32, _>(
            context,
            transfer_queue,
            command_pool,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            data,
        )
    }

    fn create_index_buffer(
        context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        data: &[u32],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        Renderer::create_device_local_buffer_with_data::<u16, _>(
            context,
            transfer_queue,
            command_pool,
            vk::BufferUsageFlags::INDEX_BUFFER,
            data,
        )
    }

    fn create_device_local_buffer_with_data<A, T: Copy>(
        context: &VkContext,
        transfer_queue: vk::Queue,
        command_pool: vk::CommandPool,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;

        let (staging_buffer, staging_buffer_memory, staging_mem_size) = Renderer::create_buffer(
            context,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        unsafe {
            let data_ptr = context
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map memory");
            let mut align = util::Align::new(data_ptr, align_of::<A>() as _, staging_mem_size);
            align.copy_from_slice(data);
            context.device.unmap_memory(staging_buffer_memory);
        };
        let (buffer, mem, _) = Renderer::create_buffer(
            context,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        Renderer::copy_buffer(
            &context.device,
            transfer_queue,
            command_pool,
            staging_buffer,
            buffer,
            buffer_size,
        );

        unsafe {
            context.device.destroy_buffer(staging_buffer, None);
            context.device.free_memory(staging_buffer_memory, None);
        };
        (buffer, mem)
    }

    fn create_command_pool(
        device: &Device,
        queue_family: &QueueFamilyIndices,
        flags: vk::CommandPoolCreateFlags,
    ) -> vk::CommandPool {
        let command_pool_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags,
            queue_family_index: queue_family.graphics_family.unwrap(),
        };
        unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .expect("Failed to create command pool")
        }
    }

    fn create_command_buffers(
        device: &Device,
        command_pool: vk::CommandPool,
        gfx_pipeline: vk::Pipeline,
        render_pass: vk::RenderPass,
        frambuffers: &Vec<vk::Framebuffer>,
        surface_extent: vk::Extent2D,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        ind_buf_len: u32,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
    ) -> Vec<vk::CommandBuffer> {
        let command_buffer_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_buffer_count: frambuffers.len() as u32,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
        };
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_info)
                .expect("Failed to allocate command buffers")
        };
        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
                p_inheritance_info: ptr::null(),
            };
            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording a command buffer");
            }
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];
            let render_pass_begin_info = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                p_next: ptr::null(),
                render_pass,
                framebuffer: frambuffers[i],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: surface_extent,
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
            };
            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    gfx_pipeline,
                );
                let vertex_buffers = [vertex_buffer];
                let offsets = [0_u64];
                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                let null = [];
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets[i..=i],
                    &null,
                );
                device.cmd_draw_indexed(command_buffer, ind_buf_len, 1, 0, 0, 0);
                device.cmd_end_render_pass(command_buffer);
                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to end record command buffer");
            }
        }
        command_buffers
    }

    fn create_buffer(
        context: &VkContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required_mem_props: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory, vk::DeviceSize) {
        let buffer_create_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::BufferCreateFlags::empty(),
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        let buffer = unsafe {
            context
                .device
                .create_buffer(&buffer_create_info, None)
                .expect("Failed to create buffer")
        };
        let mem_reqs = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let mem_type = Renderer::pick_memory_type(
            mem_reqs.memory_type_bits,
            required_mem_props,
            &context.physical_device_memory_properties,
        );
        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_reqs.size,
            memory_type_index: mem_type,
        };
        let buffer_memory = unsafe {
            context
                .device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate buffer memory")
        };
        unsafe {
            context
                .device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind buffer");
        }
        (buffer, buffer_memory, mem_reqs.size)
    }

    fn copy_buffer(
        device: &Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        let allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_buffer_count: 1,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
        };

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate command buffer")
        };
        let command_buffer = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: ptr::null(),
        };
        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer");
            let copy_regions = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }];
            device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &copy_regions);
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer");
        }
        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
        }];
        unsafe {
            device
                .queue_submit(queue, &submit_infos, vk::Fence::null())
                .expect("Failed to submit queue");
            device
                .queue_wait_idle(queue)
                .expect("Failed to wait for queue to become idle");
            device.free_command_buffers(command_pool, &command_buffers);
        }
    }

    fn create_descriptor_set_layout(
        device: &Device,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> vk::DescriptorSetLayout {
        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .expect("Failed to create descriptor set layout")
        }
    }

    fn create_descriptor_pool(device: &Device, size: u32) -> vk::DescriptorPool {
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size,
        };
        let pool_sizes = [ubo_pool_size];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            max_sets: size,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
        };
        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create descriptor pool")
        }
    }

    fn create_descriptor_sets(
        device: &Device,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
    ) -> Vec<vk::DescriptorSet> {
        let layouts = (0..uniform_buffers.len())
            .map(|_| layout)
            .collect::<Vec<vk::DescriptorSetLayout>>();
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_set_count: uniform_buffers.len() as u32,
            descriptor_pool: pool,
            p_set_layouts: layouts.as_ptr(),
        };
        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets")
        };
        descriptor_sets
            .iter()
            .zip(uniform_buffers.iter())
            .for_each(|(set, &buf)| {
                let buffer_info = vk::DescriptorBufferInfo {
                    buffer: buf,
                    offset: 0,
                    range: std::mem::size_of::<CameraUBO>() as u64, // TODO: refactor this so that its generic
                };
                let ubo_descriptor_write = [vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: *set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: &buffer_info,
                    p_texel_buffer_view: ptr::null(),
                }];
                unsafe {
                    device.update_descriptor_sets(&ubo_descriptor_write, &[]);
                }
            });
        descriptor_sets
    }

    fn create_sync_objects(device: &Device) -> SyncObjects {
        let mut sync_objects = SyncObjects {
            image_available_semaphores: vec![],
            render_finished_semaphores: vec![],
            inflight_fences: vec![],
        };
        let sem_create_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
        };
        let fen_create_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
        };
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                let image_available_semaphore = device
                    .create_semaphore(&sem_create_info, None)
                    .expect("Fauled to create image available semaphore");
                let render_finished_semaphore = device
                    .create_semaphore(&sem_create_info, None)
                    .expect("Failed to create render finished semaphore");
                let inflight_fence = device
                    .create_fence(&fen_create_info, None)
                    .expect("Failed to craete inflight fence");
                sync_objects
                    .image_available_semaphores
                    .push(image_available_semaphore);
                sync_objects
                    .render_finished_semaphores
                    .push(render_finished_semaphore);
                sync_objects.inflight_fences.push(inflight_fence);
            }
        }
        sync_objects
    }

    fn setup_debug_utils(
        entry: &Entry,
        instance: &Instance,
    ) -> (extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT) {
        let debug_utils_loader = extensions::ext::DebugUtils::new(entry, instance);
        if VALIDATION.is_enabled == false {
            (debug_utils_loader, vk::DebugUtilsMessengerEXT::null())
        } else {
            let messenger_ci = populate_debug_messenger_create_info();
            let messenger = unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&messenger_ci, None)
                    .expect("Failed to load debug utils")
            };
            (debug_utils_loader, messenger)
        }
    }

    fn pick_suitable_physical_device(
        instance: &Instance,
        surface_data: &SurfaceData,
    ) -> vk::PhysicalDevice {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };
        println!("Found {} physical devices", physical_devices.len());
        let mut result = None;
        for &device in physical_devices.iter() {
            if queries::is_physical_device_suitable(
                instance,
                device,
                surface_data,
                &DEVICE_EXTENSIONS,
            ) {
                if result == None {
                    result = Some(device)
                }
            }
        }
        match result {
            None => panic!("Failed to find a suitable GPU"),
            Some(device) => device,
        }
    }

    fn pick_swapchain_format(formats: &Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
        *formats
            .iter()
            .find(|&&fmt| {
                fmt.format == vk::Format::B8G8R8A8_SRGB
                    && fmt.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&formats.first().unwrap().clone())
    }

    fn pick_swapchain_present_mode(
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        *available_present_modes
            .iter()
            .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(&vk::PresentModeKHR::FIFO)
    }

    fn pick_swapchain_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> ash::vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: WINDOW_WIDTH.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: WINDOW_HEIGHT.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn pick_memory_type(
        type_filter: u32,
        required_properties: vk::MemoryPropertyFlags,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> u32 {
        for (i, mem_type) in memory_properties.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) > 0 && mem_type.property_flags.contains(required_properties)
            {
                return i as u32;
            }
        }
        panic!("Failed to find suitable memory");
    }

    fn drop_swapchain(&self) {
        unsafe {
            self.context
                .device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            for &framebuffer in self.framebuffers.iter() {
                self.context.device.destroy_framebuffer(framebuffer, None);
            }
            self.context
                .device
                .destroy_pipeline(self.gfx_pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.context
                .device
                .destroy_render_pass(self.render_pass, None);
            for &imageview in self.swapchain_imageviews.iter() {
                self.context.device.destroy_image_view(imageview, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }

    fn create_uniform_buffer<T>(
        context: &VkContext,
        len: usize,
    ) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
        let size = std::mem::size_of::<T>() as vk::DeviceSize;
        let mut buffers = Vec::new();
        let mut mems = Vec::new();
        for _ in 0..len {
            let (buf, mem, _) = Renderer::create_buffer(
                context,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffers.push(buf);
            mems.push(mem);
        }
        (buffers, mems)
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.context
                    .device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.context
                    .device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.context
                    .device
                    .destroy_fence(self.in_flight_fences[i], None);
                self.context.device.destroy_buffer(self.ubos[i], None);
                self.context.device.free_memory(self.ubo_mems[i], None);
            }
            self.context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.context
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.drop_swapchain();
            self.context.device.destroy_buffer(self.vertex_buffer, None);
            self.context
                .device
                .free_memory(self.vertex_buffer_mem, None);
            self.context.device.destroy_buffer(self.index_buffer, None);
            self.context.device.free_memory(self.index_buffer_mem, None);
            self.context
                .device
                .destroy_command_pool(self.command_pool_transient, None);
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
            if VALIDATION.is_enabled {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub fn required_instance_extension_names() -> Vec<*const i8> {
    vec![
        extensions::khr::Surface::name().as_ptr(),
        extensions::mvk::MacOSSurface::name().as_ptr(),
        extensions::ext::DebugUtils::name().as_ptr(),
        vk::KhrPortabilityEnumerationFn::name().as_ptr(),
        vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
    ]
}

#[cfg(target_os = "macos")]
pub fn required_device_extension_names() -> Vec<*const i8> {
    vec![
        vk::KhrPortabilitySubsetFn::name().as_ptr(),
        vk::KhrSwapchainFn::name().as_ptr(),
    ]
}

#[cfg(all(windows))]
pub fn required_device_extension_names() -> Vec<*const i8> {
    vec![vk::KhrSwapchainFn::name().as_ptr()]
}

#[cfg(all(unix, not(target_os = "macos")))]
pub fn required_device_extension_names() -> Vec<*const i8> {
    vec![vk::KhrSwapchainFn::name().as_ptr()]
}

#[cfg(all(windows))]
pub fn required_instance_extension_names() -> Vec<*const i8> {
    vec![
        extensions::khr::Surface::name().as_ptr(),
        extensions::khr::Win32Surface::name().as_ptr(),
        extensions::ext::DebugUtils::name().as_ptr(),
    ]
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_instance_extension_names() -> Vec<*const i8> {
    vec![
        extensions::khr::Surface::name().as_ptr(),
        extensions::khr::XlibSurface::name().as_ptr(),
        extensions::ext::DebugUtils::name().as_ptr(),
    ]
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, ash::vk::Result> {
    use winit::platform::x11::WindowExtX11;
    let x11_display = window.xlib_display().unwrap();
    let x11_window = window.xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR {
        s_type: vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        window: x11_window as vk::Window,
        dpy: x11_display as *mut vk::Display,
    };
    let xlib_surface_loader = extensions::khr::XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

#[cfg(target_os = "macos")]
pub unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, ash::vk::Result> {
    use cocoa::appkit::{NSView, NSWindow};
    use cocoa::base::id as cocoa_id;
    use objc::runtime::YES;
    use std::mem;
    use winit::platform::macos::WindowExtMacOS;
    let wnd: cocoa_id = mem::transmute(window.ns_window());

    let layer = metal::MetalLayer::new();

    layer.set_edge_antialiasing_mask(0);
    layer.set_presents_with_transaction(false);
    layer.remove_all_animations();

    let view = wnd.contentView();

    layer.set_contents_scale(view.backingScaleFactor());
    view.setLayer(mem::transmute(layer.as_ref()));
    view.setWantsLayer(YES);

    let create_info = vk::MacOSSurfaceCreateInfoMVK {
        s_type: vk::StructureType::MACOS_SURFACE_CREATE_INFO_MVK,
        p_next: ptr::null(),
        flags: Default::default(),
        p_view: window.ns_view() as *const c_void,
    };

    let macos_surface_loader = extensions::mvk::MacOSSurface::new(entry, instance);
    macos_surface_loader.create_mac_os_surface(&create_info, None)
}

#[cfg(target_os = "windows")]
pub unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, ash::vk::Result> {
    use winapi::shared::windef::HWND;
    use winapi::um::libloaderapi::GetModuleHandleW;
    use winit::platform::windows::WindowExtWindows;

    let hwnd = window.hwnd() as HWND;
    let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
        s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        hinstance,
        hwnd: hwnd as *const c_void,
    };
    let win32_surface_loader = extensions::khr::Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}
