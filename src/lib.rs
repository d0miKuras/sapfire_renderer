use std::{ffi::CString, os::raw::c_void, ptr};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
mod debug;
mod helpers;
use debug::{populate_debug_messenger_create_info, ValidationInfo};
use helpers::vk_to_string;

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

pub const VALIDATION: ValidationInfo = ValidationInfo {
    is_enabled: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};

struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
}

pub struct Renderer {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: ash::vk::DebugUtilsMessengerEXT,
    _gpu: ash::vk::PhysicalDevice,
    device: ash::Device,
    _gfx_queue: ash::vk::Queue,
}

impl Renderer {
    pub fn new() -> Renderer {
        // init vulkan stuff
        unsafe {
            let entry = ash::Entry::load().unwrap();
            if VALIDATION.is_enabled && !Renderer::check_validation_layer_support(&entry) {
                panic!("Validation layer requested but it's not available");
            }
            let instance = Renderer::create_instance(&entry);
            let (debug_utils_loader, debug_messenger) =
                Renderer::setup_debug_utils(&entry, &instance);
            let gpu = Renderer::pick_suitable_physical_device(&instance);
            let (device, gfx_queue) = Renderer::create_logical_device(&instance, gpu);
            Renderer {
                _entry: entry,
                instance,
                debug_utils_loader,
                debug_messenger,
                _gpu: gpu,
                device,
                _gfx_queue: gfx_queue,
            }
        }
    }

    pub fn create_instance(entry: &ash::Entry) -> ash::Instance {
        let app_info = ash::vk::ApplicationInfo {
            api_version: ash::vk::API_VERSION_1_0,
            engine_version: 0,
            s_type: ash::vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: CString::new("Sapfire").unwrap().as_ptr(),
            p_engine_name: CString::new("Sapfire Engine").unwrap().as_ptr(),
            application_version: 0,
        };
        let debug_utils_create_info = populate_debug_messenger_create_info();
        let extension_names = required_extension_names();
        let required_validation_names_raw: Vec<CString> = VALIDATION
            .required_validation_layers
            .iter()
            .map(|name| CString::new(*name).unwrap())
            .collect();
        let enabled_layer_names: Vec<*const i8> = required_validation_names_raw
            .iter()
            .map(|name| name.as_ptr())
            .collect();
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
        let instance: ash::Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create Vulkan instance!")
        };
        instance
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

            _ => (),
        })
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: ash::vk::PhysicalDevice,
    ) -> (ash::Device, ash::vk::Queue) {
        let indices = Renderer::find_queue_family(instance, physical_device);
        let q_prios = [1.0_f32];
        let extension_names = required_extension_names_queue();
        let queue_create_info = ash::vk::DeviceQueueCreateInfo {
            s_type: ash::vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            flags: ash::vk::DeviceQueueCreateFlags::empty(),
            p_next: ptr::null(),
            p_queue_priorities: q_prios.as_ptr(),
            queue_count: q_prios.len() as u32,
            queue_family_index: indices.graphics_family.unwrap(),
        };
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
            p_queue_create_infos: &queue_create_info,
            queue_create_info_count: 1,
            p_next: ptr::null(),
            flags: ash::vk::DeviceCreateFlags::default(),
        };
        let device: ash::Device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };
        let graphics_q = unsafe { device.get_device_queue(indices.graphics_family.unwrap(), 0) };
        (device, graphics_q)
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

    fn pick_suitable_physical_device(instance: &ash::Instance) -> ash::vk::PhysicalDevice {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };
        println!("Found {} physical devices", physical_devices.len());
        let mut result = None;
        for &device in physical_devices.iter() {
            if Renderer::is_physical_device_suitable(instance, device) {
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

    fn check_validation_layer_support(entry: &ash::Entry) -> bool {
        let layer_props = entry
            .enumerate_instance_layer_properties()
            .expect("Failed to enumerate layer instance properties");
        if layer_props.len() <= 0 {
            eprintln!("No available layer properties");
            return false;
        } else {
            for prop in layer_props.iter() {
                println!("{}", vk_to_string(&prop.layer_name));
            }
        }
        for req_layer_name in VALIDATION.required_validation_layers.iter() {
            let mut layer_found = false;
            for prop in layer_props.iter() {
                let layer_name = vk_to_string(&prop.layer_name);
                if (*req_layer_name) == layer_name {
                    layer_found = true;
                    break;
                }
            }
            if layer_found == false {
                return false;
            }
        }
        true
    }

    fn is_physical_device_suitable(
        instance: &ash::Instance,
        physical_device: ash::vk::PhysicalDevice,
    ) -> bool {
        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_features = unsafe { instance.get_physical_device_features(physical_device) };
        let device_queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let device_type = match device_properties.device_type {
            ash::vk::PhysicalDeviceType::CPU => "Cpu",
            ash::vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
            ash::vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
            ash::vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
            ash::vk::PhysicalDeviceType::OTHER => "Unknown",
            _ => panic!(),
        };
        let device_name = vk_to_string(&device_properties.device_name);
        println!(
            "\tDevice Name: {}, id: {}, type: {}",
            device_name, device_properties.device_id, device_type
        );
        let vmajor = ash::vk::api_version_major(device_properties.api_version);
        let vminor = ash::vk::api_version_minor(device_properties.api_version);
        let vpatch = ash::vk::api_version_patch(device_properties.api_version);
        println!("\tAPI Version: {}.{}.{}", vmajor, vminor, vpatch);
        println!("\tYES Queue Family: {}", device_queue_families.len());
        println!("\t\tQueue Count | Graphics, Compute, Transfer, Sparse Binding");
        for queue_family in device_queue_families.iter() {
            let is_graphics_support = if queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::GRAPHICS)
            {
                "YES"
            } else {
                "NO"
            };
            let is_compute_support = if queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::COMPUTE)
            {
                "YES"
            } else {
                "NO"
            };
            let is_transfer_support = if queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::TRANSFER)
            {
                "YES"
            } else {
                "NO"
            };
            let is_sparse_support = if queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::SPARSE_BINDING)
            {
                "YES"
            } else {
                "NO"
            };
            println!(
                "\t\t{}\t    | {},\t  {},\t  {},\t  {}",
                queue_family.queue_count,
                is_graphics_support,
                is_compute_support,
                is_transfer_support,
                is_sparse_support
            );
        }
        println!(
            "\tGeometry Shader YES: {}",
            if device_features.geometry_shader == 1 {
                "YES"
            } else {
                "NO"
            }
        );
        let indices = Renderer::find_queue_family(instance, physical_device);

        return indices.graphics_family.is_some();
    }

    fn find_queue_family(
        instance: &ash::Instance,
        physical_device: ash::vk::PhysicalDevice,
    ) -> QueueFamilyIndices {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let mut queue_family_indices = QueueFamilyIndices {
            graphics_family: None,
        };
        let mut index = 0;
        for fam in queue_families {
            if fam.queue_count > 0 && fam.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) {
                queue_family_indices.graphics_family = Some(index);
                break;
            }
            index += 1;
        }
        queue_family_indices
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            if VALIDATION.is_enabled {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(target_os = "macos")]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        ash::extensions::khr::Surface::name().as_ptr(),
        ash::extensions::mvk::MacOSSurface::name().as_ptr(),
        ash::extensions::ext::DebugUtils::name().as_ptr(),
        ash::vk::KhrPortabilityEnumerationFn::name().as_ptr(),
        ash::vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
    ]
}

#[cfg(target_os = "macos")]
pub fn required_extension_names_queue() -> Vec<*const i8> {
    vec![ash::vk::KhrPortabilitySubsetFn::name().as_ptr()]
}

#[cfg(all(windows))]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        Win32Surface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}
