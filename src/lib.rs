use ash::extensions;
use ash::vk;
use ash::Device;
use ash::Entry;
use ash::Instance;
use debug::{populate_debug_messenger_create_info, ValidationInfo};
use queries::QueueFamilyIndices;
use std::{ffi::CString, os::raw::c_void, ptr};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
mod debug;
mod helpers;
mod queries;

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const MAX_FRAMES_IN_FLIGHT: usize = 2;
const VERTEX_SHADER: &str = "
#version 450

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
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
    _entry: Entry,
    instance: Instance,
    surface: vk::SurfaceKHR,
    surface_loader: extensions::khr::Surface,
    debug_utils_loader: extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    _gpu: vk::PhysicalDevice,
    device: Device,
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
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
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
            let gfx_queue = device.get_device_queue(indices.graphics_family.unwrap(), 0);
            let present_queue = device.get_device_queue(indices.present_family.unwrap(), 0);
            let swapchain_data =
                Renderer::create_swap_chain(&instance, &device, gpu, &surface_data, &indices);
            let swapchain_imageviews = Renderer::create_image_views(
                &device,
                swapchain_data.swapchain_format,
                &swapchain_data.swapchain_images,
            );
            let render_pass =
                Renderer::create_render_pass(&device, swapchain_data.swapchain_format);
            let (gfx_pipeline, pipeline_layout) = Renderer::create_graphics_pipeline(
                &device,
                render_pass,
                swapchain_data.swapchain_extent,
            );
            let framebuffers = Renderer::create_framebuffers(
                &device,
                render_pass,
                &swapchain_imageviews,
                &swapchain_data.swapchain_extent,
            );
            let command_pool = Renderer::create_command_pool(&device, &indices);
            let command_buffers = Renderer::create_command_buffers(
                &device,
                command_pool,
                gfx_pipeline,
                render_pass,
                &framebuffers,
                swapchain_data.swapchain_extent,
            );
            let sync_object = Renderer::create_sync_objects(&device);
            Renderer {
                _entry: entry,
                instance,
                debug_utils_loader,
                debug_messenger,
                _gpu: gpu,
                device,
                gfx_queue,
                present_queue,
                surface: surface_data.surface,
                surface_loader: surface_data.surface_loader,
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
                command_buffers,
                image_available_semaphores: sync_object.image_available_semaphores,
                render_finished_semaphores: sync_object.render_finished_semaphores,
                in_flight_fences: sync_object.inflight_fences,
                current_frame: 0,
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
                _ => {}
            },
            Event::RedrawEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_windowid) => self.draw_frame(),
            Event::LoopDestroyed => unsafe {
                self.device
                    .device_wait_idle()
                    .expect("Failed to wait for device to become idle");
            },

            _ => (),
        })
    }

    fn draw_frame(&mut self) {
        let wait_fences = [self.in_flight_fences[self.current_frame]];
        let (next_image_index, _) = unsafe {
            self.device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for fences");
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    std::u64::MAX,
                    self.image_available_semaphores[self.current_frame],
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next framebuffer")
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
            p_command_buffers: &self.command_buffers[next_image_index as usize],
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        }];
        unsafe {
            self.device
                .reset_fences(&wait_fences)
                .expect("Failed to reset wait fences");
            self.device
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
            p_image_indices: &next_image_index,
            p_results: ptr::null_mut(),
        };
        unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
                .expect("Failed to submit present queue");
        }
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
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
        }
    }

    fn create_swap_chain(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        surface_data: &SurfaceData,
        queue_family: &QueueFamilyIndices,
    ) -> SwapChainData {
        let swapchain_support = queries::query_swapchain_support(physical_device, surface_data);
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
            surface: surface_data.surface,
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
        let swapchain_loader = extensions::khr::Swapchain::new(instance, device);
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
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            // since I set the vertex data directly in the shader
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: ptr::null(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: ptr::null(),
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
            set_layout_count: 0,
            p_set_layouts: ptr::null(),
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

    fn create_command_pool(device: &Device, queue_family: &QueueFamilyIndices) -> vk::CommandPool {
        let command_pool_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::CommandPoolCreateFlags::empty(),
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
                device.cmd_draw(command_buffer, 3, 1, 0, 0);
                device.cmd_end_render_pass(command_buffer);
                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to end record command buffer");
            }
        }
        command_buffers
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
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain_imageviews
                .iter()
                .for_each(|&view| self.device.destroy_image_view(view, None));
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_pipeline(self.gfx_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_device(None);
            if VALIDATION.is_enabled {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
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

#[cfg(all(linux))]
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
