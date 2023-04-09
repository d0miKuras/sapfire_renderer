use std::{ffi::CString, ptr};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

pub struct Renderer {
    _entry: ash::Entry,
    instance: ash::Instance,
}

impl Renderer {
    pub fn new() -> Renderer {
        // init vulkan stuff
        unsafe {
            let entry = ash::Entry::load().unwrap();
            let instance = Renderer::create_instance(&entry);
            Renderer {
                _entry: entry,
                instance,
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
        let extension_names = required_extension_names();
        let create_info = ash::vk::InstanceCreateInfo {
            enabled_extension_count: extension_names.len() as u32,
            pp_enabled_extension_names: extension_names.as_ptr(),
            s_type: ash::vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: ash::vk::InstanceCreateFlags::empty(),
            pp_enabled_layer_names: ptr::null(),
            p_application_info: &app_info,
            enabled_layer_count: 0,
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
                            dbg!();
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
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
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
