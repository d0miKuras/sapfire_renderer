use ash::vk;
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
    pub surface: ash::vk::SurfaceKHR,
    pub surface_loader: ash::extensions::khr::Surface,
}

struct SwapChainData {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: ash::vk::SwapchainKHR,
    swapchain_images: Vec<ash::vk::Image>,
    swapchain_format: ash::vk::Format,
    swapchain_extent: ash::vk::Extent2D,
}

pub struct Renderer {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface: ash::vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: ash::vk::DebugUtilsMessengerEXT,
    _gpu: ash::vk::PhysicalDevice,
    device: ash::Device,
    _gfx_queue: ash::vk::Queue,
    _present_queue: ash::vk::Queue,
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: ash::vk::SwapchainKHR,
    swapchain_images: Vec<ash::vk::Image>,
    swapchain_format: ash::vk::Format,
    swapchain_extent: ash::vk::Extent2D,
    swapchain_imageviews: Vec<ash::vk::ImageView>,
    render_pass: ash::vk::RenderPass,
    pipeline_layout: ash::vk::PipelineLayout,
    gfx_pipeline: ash::vk::Pipeline,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Renderer {
        // init vulkan stuff
        unsafe {
            let entry = ash::Entry::load().unwrap();
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
            Renderer {
                _entry: entry,
                instance,
                debug_utils_loader,
                debug_messenger,
                _gpu: gpu,
                device,
                _gfx_queue: gfx_queue,
                _present_queue: present_queue,
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

    pub fn main_loop(self, window: winit::window::Window, event_loop: EventLoop<()>) {
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

            _ => (),
        })
    }

    fn create_instance(entry: &ash::Entry) -> ash::Instance {
        let application_name = CString::new("Sapfire").unwrap();
        let engine_name = CString::new("Sapfire Engine").unwrap();
        let app_info = ash::vk::ApplicationInfo {
            api_version: ash::vk::API_VERSION_1_0,
            engine_version: 0,
            s_type: ash::vk::StructureType::APPLICATION_INFO,
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
            let create_info = ash::vk::InstanceCreateInfo {
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
                    &debug_utils_create_info as *const ash::vk::DebugUtilsMessengerCreateInfoEXT
                        as *const c_void
                } else {
                    ptr::null()
                },
                s_type: ash::vk::StructureType::INSTANCE_CREATE_INFO,
                flags: ash::vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
                p_application_info: &app_info,
            };
        }
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        {
            let create_info = ash::vk::InstanceCreateInfo {
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
                    &debug_utils_create_info as *const ash::vk::DebugUtilsMessengerCreateInfoEXT
                        as *const c_void
                } else {
                    ptr::null()
                },
                s_type: ash::vk::StructureType::INSTANCE_CREATE_INFO,
                flags: ash::vk::InstanceCreateFlags::empty(),
                p_application_info: &app_info,
            };
            let instance: ash::Instance = unsafe {
                entry
                    .create_instance(&create_info, None)
                    .expect("Failed to create Vulkan instance!")
            };
            instance
        }
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: ash::vk::PhysicalDevice,
        surface_data: &SurfaceData,
    ) -> (ash::Device, QueueFamilyIndices) {
        let indices = queries::find_queue_family(instance, physical_device, surface_data);
        let mut unique_families = std::collections::HashSet::new();
        unique_families.insert(indices.graphics_family);
        unique_families.insert(indices.present_family);
        let q_prios = [1.0_f32];
        let extension_names = required_device_extension_names();
        let mut q_create_infos = vec![];
        for &q_fam in unique_families.iter() {
            let queue_create_info = ash::vk::DeviceQueueCreateInfo {
                s_type: ash::vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                flags: ash::vk::DeviceQueueCreateFlags::empty(),
                p_next: ptr::null(),
                p_queue_priorities: q_prios.as_ptr(),
                queue_count: q_prios.len() as u32,
                queue_family_index: q_fam.unwrap(),
            };
            q_create_infos.push(queue_create_info);
        }

        let gpu_device_features = ash::vk::PhysicalDeviceFeatures {
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
        let device_create_info = ash::vk::DeviceCreateInfo {
            s_type: ash::vk::StructureType::DEVICE_CREATE_INFO,
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
            flags: ash::vk::DeviceCreateFlags::empty(),
        };
        let device: ash::Device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };
        (device, indices)
    }

    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> SurfaceData {
        let surface = unsafe { create_surface(entry, instance, window).unwrap() };
        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);
        SurfaceData {
            surface,
            surface_loader,
        }
    }

    fn create_swap_chain(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: ash::vk::PhysicalDevice,
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
                    ash::vk::SharingMode::CONCURRENT,
                    2,
                    vec![
                        queue_family.graphics_family.unwrap(),
                        queue_family.present_family.unwrap(),
                    ],
                )
            } else {
                (ash::vk::SharingMode::EXCLUSIVE, 0, vec![])
            };
        let swapchain_create_info = ash::vk::SwapchainCreateInfoKHR {
            s_type: ash::vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            surface: surface_data.surface,
            min_image_count: swapchain_support.capabilities.min_image_count + 1,
            p_next: ptr::null(),
            flags: ash::vk::SwapchainCreateFlagsKHR::empty(),
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: ash::vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            queue_family_index_count,
            p_queue_family_indices: queue_family_indexes.as_ptr(),
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: ash::vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: ash::vk::TRUE,
            old_swapchain: ash::vk::SwapchainKHR::null(),
        };
        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device);
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
        device: &ash::Device,
        swapchain_format: ash::vk::Format,
        swapchain_images: &Vec<ash::vk::Image>,
    ) -> Vec<ash::vk::ImageView> {
        let mut image_views = vec![];
        for &image in swapchain_images {
            let imageview_create_info = ash::vk::ImageViewCreateInfo {
                s_type: ash::vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: ash::vk::ImageViewCreateFlags::empty(),
                image,
                format: swapchain_format,
                view_type: ash::vk::ImageViewType::TYPE_2D,
                components: ash::vk::ComponentMapping {
                    a: ash::vk::ComponentSwizzle::IDENTITY,
                    r: ash::vk::ComponentSwizzle::IDENTITY,
                    g: ash::vk::ComponentSwizzle::IDENTITY,
                    b: ash::vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: ash::vk::ImageSubresourceRange {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
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

    fn create_render_pass(
        device: &ash::Device,
        swapchain_format: ash::vk::Format,
    ) -> ash::vk::RenderPass {
        let color_attachment = ash::vk::AttachmentDescription {
            format: swapchain_format,
            flags: ash::vk::AttachmentDescriptionFlags::empty(),
            samples: ash::vk::SampleCountFlags::TYPE_1,
            load_op: ash::vk::AttachmentLoadOp::CLEAR,
            store_op: ash::vk::AttachmentStoreOp::STORE,
            stencil_load_op: ash::vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: ash::vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: ash::vk::ImageLayout::UNDEFINED,
            final_layout: ash::vk::ImageLayout::PRESENT_SRC_KHR,
        };
        let color_attachment_ref = ash::vk::AttachmentReference {
            attachment: 0,
            layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let subpass_descr = ash::vk::SubpassDescription {
            flags: ash::vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: ash::vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_resolve_attachments: ptr::null(),
            p_depth_stencil_attachment: ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        };
        let renderpass_create_info = ash::vk::RenderPassCreateInfo {
            s_type: ash::vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::RenderPassCreateFlags::empty(),
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass_descr,
            dependency_count: 0,
            p_dependencies: ptr::null(),
        };
        unsafe {
            device
                .create_render_pass(&renderpass_create_info, None)
                .expect("Failed to create render pass")
        }
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: ash::vk::RenderPass,
        swapchain_extent: ash::vk::Extent2D,
    ) -> (ash::vk::Pipeline, ash::vk::PipelineLayout) {
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
            ash::vk::PipelineShaderStageCreateInfo {
                s_type: ash::vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                stage: ash::vk::ShaderStageFlags::VERTEX,
                module: vert_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
            },
            ash::vk::PipelineShaderStageCreateInfo {
                s_type: ash::vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                stage: ash::vk::ShaderStageFlags::FRAGMENT,
                module: frag_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
            },
        ];
        let dynamic_states = [
            ash::vk::DynamicState::VIEWPORT,
            ash::vk::DynamicState::SCISSOR,
        ];
        let dynamic_state_info = ash::vk::PipelineDynamicStateCreateInfo {
            s_type: ash::vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::PipelineDynamicStateCreateFlags::empty(),
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
        };
        let vertex_input_info = ash::vk::PipelineVertexInputStateCreateInfo {
            // since I set the vertex data directly in the shader
            s_type: ash::vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: ptr::null(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: ptr::null(),
        };
        let input_assembly_info = ash::vk::PipelineInputAssemblyStateCreateInfo {
            s_type: ash::vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::PipelineInputAssemblyStateCreateFlags::empty(),
            topology: ash::vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: ash::vk::FALSE,
        };
        let viewports = [ash::vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [ash::vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        }];
        let viewport_state_info = ash::vk::PipelineViewportStateCreateInfo {
            s_type: ash::vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::PipelineViewportStateCreateFlags::empty(),
            viewport_count: 1,
            p_viewports: viewports.as_ptr(),
            scissor_count: 1,
            p_scissors: scissors.as_ptr(),
        };
        let rasterizer_info = ash::vk::PipelineRasterizationStateCreateInfo {
            s_type: ash::vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: ash::vk::FALSE, // will clamp depth value instead of discarding on fragments beyond near and far planes. Useful in shadowmapping.
            rasterizer_discard_enable: ash::vk::FALSE, // TRUE will not allow geometry to pass through rasterizer stage
            polygon_mode: ash::vk::PolygonMode::FILL,
            cull_mode: ash::vk::CullModeFlags::BACK,
            front_face: ash::vk::FrontFace::CLOCKWISE,
            depth_bias_enable: ash::vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };
        let multisampling = ash::vk::PipelineMultisampleStateCreateInfo {
            // should be turned off for now
            s_type: ash::vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::PipelineMultisampleStateCreateFlags::empty(),
            rasterization_samples: ash::vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: ash::vk::FALSE,
            min_sample_shading: 1.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: ash::vk::FALSE,
            alpha_to_one_enable: ash::vk::FALSE,
        };
        let color_blend_attachment = [ash::vk::PipelineColorBlendAttachmentState {
            // for alpha blending = bad performance, so turned off
            color_write_mask: ash::vk::ColorComponentFlags::RGBA,
            blend_enable: ash::vk::FALSE,
            src_color_blend_factor: ash::vk::BlendFactor::ONE,
            dst_color_blend_factor: ash::vk::BlendFactor::ZERO,
            color_blend_op: ash::vk::BlendOp::ADD,
            src_alpha_blend_factor: ash::vk::BlendFactor::ONE,
            dst_alpha_blend_factor: ash::vk::BlendFactor::ZERO,
            alpha_blend_op: ash::vk::BlendOp::ADD,
        }];
        let color_blend_info = ash::vk::PipelineColorBlendStateCreateInfo {
            s_type: ash::vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: ash::vk::FALSE,
            logic_op: ash::vk::LogicOp::COPY,
            attachment_count: color_blend_attachment.len() as u32,
            p_attachments: color_blend_attachment.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };
        let pipeline_layout_info = ash::vk::PipelineLayoutCreateInfo {
            s_type: ash::vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::PipelineLayoutCreateFlags::empty(),
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
        let pipeline_infos = [ash::vk::GraphicsPipelineCreateInfo {
            s_type: ash::vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::PipelineCreateFlags::empty(),
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
            p_dynamic_state: &dynamic_state_info,
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            base_pipeline_handle: ash::vk::Pipeline::null(),
            base_pipeline_index: -1,
        }];
        let gfx_pipelines = unsafe {
            device
                .create_graphics_pipelines(ash::vk::PipelineCache::null(), &pipeline_infos, None)
                .expect("Failed to create graphics pipeline")
        };
        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }
        (gfx_pipelines[0], pipeline_layout)
    }

    fn create_shader_module(device: &ash::Device, code: Vec<u8>) -> ash::vk::ShaderModule {
        let shader_module_create_info = ash::vk::ShaderModuleCreateInfo {
            flags: ash::vk::ShaderModuleCreateFlags::empty(),
            p_next: ptr::null(),
            s_type: ash::vk::StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: code.len(),
            p_code: code.as_ptr() as *const u32,
        };
        unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create shader module")
        }
    }

    fn setup_debug_utils(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> (
        ash::extensions::ext::DebugUtils,
        ash::vk::DebugUtilsMessengerEXT,
    ) {
        let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, instance);
        if VALIDATION.is_enabled == false {
            (debug_utils_loader, ash::vk::DebugUtilsMessengerEXT::null())
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
        instance: &ash::Instance,
        surface_data: &SurfaceData,
    ) -> ash::vk::PhysicalDevice {
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

    fn pick_swapchain_format(
        formats: &Vec<ash::vk::SurfaceFormatKHR>,
    ) -> ash::vk::SurfaceFormatKHR {
        *formats
            .iter()
            .find(|&&fmt| {
                fmt.format == ash::vk::Format::B8G8R8A8_SRGB
                    && fmt.color_space == ash::vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&formats.first().unwrap().clone())
    }

    fn pick_swapchain_present_mode(
        available_present_modes: &Vec<ash::vk::PresentModeKHR>,
    ) -> ash::vk::PresentModeKHR {
        *available_present_modes
            .iter()
            .find(|&&mode| mode == ash::vk::PresentModeKHR::MAILBOX)
            .unwrap_or(&ash::vk::PresentModeKHR::FIFO)
    }

    fn pick_swapchain_extent(capabilities: &ash::vk::SurfaceCapabilitiesKHR) -> ash::vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            ash::vk::Extent2D {
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
        ash::extensions::khr::Surface::name().as_ptr(),
        ash::extensions::mvk::MacOSSurface::name().as_ptr(),
        ash::extensions::ext::DebugUtils::name().as_ptr(),
        ash::vk::KhrPortabilityEnumerationFn::name().as_ptr(),
        ash::vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
    ]
}

#[cfg(target_os = "macos")]
pub fn required_device_extension_names() -> Vec<*const i8> {
    vec![
        ash::vk::KhrPortabilitySubsetFn::name().as_ptr(),
        ash::vk::KhrSwapchainFn::name().as_ptr(),
    ]
}

#[cfg(all(windows))]
pub fn required_device_extension_names() -> Vec<*const i8> {
    vec![ash::vk::KhrSwapchainFn::name().as_ptr()]
}

#[cfg(all(unix))]
pub fn required_device_extension_names() -> Vec<*const i8> {
    vec![ash::vk::KhrSwapchainFn::name().as_ptr()]
}

#[cfg(all(windows))]
pub fn required_instance_extension_names() -> Vec<*const i8> {
    vec![
        ash::extensions::khr::Surface::name().as_ptr(),
        ash::extensions::khr::Win32Surface::name().as_ptr(),
        ash::extensions::ext::DebugUtils::name().as_ptr(),
    ]
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_instance_extension_names() -> Vec<*const i8> {
    vec![
        ash::extensions::khr::Surface::name().as_ptr(),
        ash::extensions::khr::XlibSurface::name().as_ptr(),
        ash::extensions::ext::DebugUtils::name().as_ptr(),
    ]
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<ash::vk::SurfaceKHR, ash::vk::Result> {
    use winit::platform::x11::WindowExtX11;
    let x11_display = window.xlib_display().unwrap();
    let x11_window = window.xlib_window().unwrap();
    let x11_create_info = ash::vk::XlibSurfaceCreateInfoKHR {
        s_type: ash::vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        window: x11_window as ash::vk::Window,
        dpy: x11_display as *mut ash::vk::Display,
    };
    let xlib_surface_loader = ash::extensions::khr::XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

#[cfg(target_os = "macos")]
pub unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<ash::vk::SurfaceKHR, ash::vk::Result> {
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

    let create_info = ash::vk::MacOSSurfaceCreateInfoMVK {
        s_type: ash::vk::StructureType::MACOS_SURFACE_CREATE_INFO_MVK,
        p_next: ptr::null(),
        flags: Default::default(),
        p_view: window.ns_view() as *const c_void,
    };

    let macos_surface_loader = ash::extensions::mvk::MacOSSurface::new(entry, instance);
    macos_surface_loader.create_mac_os_surface(&create_info, None)
}

#[cfg(target_os = "windows")]
pub unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<ash::vk::SurfaceKHR, ash::vk::Result> {
    use winapi::shared::windef::HWND;
    use winapi::um::libloaderapi::GetModuleHandleW;
    use winit::platform::windows::WindowExtWindows;

    let hwnd = window.hwnd() as HWND;
    let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
    let win32_create_info = ash::vk::Win32SurfaceCreateInfoKHR {
        s_type: ash::vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        hinstance,
        hwnd: hwnd as *const c_void,
    };
    let win32_surface_loader = ash::extensions::khr::Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}
