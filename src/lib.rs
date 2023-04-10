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

pub struct Renderer {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: ash::vk::DebugUtilsMessengerEXT,
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
            Renderer {
                _entry: entry,
                instance,
                debug_utils_loader,
                debug_messenger,
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

    pub fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title("Sapfire Renderer")
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(event_loop)
            .expect("Failed to create window.")
    }

    pub fn main_loop(window: winit::window::Window, event_loop: EventLoop<()>) {
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
    ]
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
