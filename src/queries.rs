use crate::{helpers::vk_to_string, Renderer, VALIDATION};
pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub present_family: Option<u32>,
}

pub struct SurfaceData {
    pub surface: ash::vk::SurfaceKHR,
    pub surface_loader: ash::extensions::khr::Surface,
}

pub fn check_validation_layer_support(entry: &ash::Entry) -> bool {
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

pub fn is_physical_device_suitable(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
    surface_data: &SurfaceData,
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
    println!("\tSupports Queue Families: {}", device_queue_families.len());
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
        "\tGeometry Shader Support: {}",
        if device_features.geometry_shader == 1 {
            "YES"
        } else {
            "NO"
        }
    );
    let indices = find_queue_family(instance, physical_device, surface_data);

    return indices.graphics_family.is_some();
}

pub fn find_queue_family(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
    surface_data: &SurfaceData,
) -> QueueFamilyIndices {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let mut queue_family_indices = QueueFamilyIndices {
        graphics_family: None,
        present_family: None,
    };
    let mut index = 0;
    for fam in queue_families {
        if fam.queue_count > 0 && fam.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) {
            queue_family_indices.graphics_family = Some(index);
        }
        let present_supported = unsafe {
            surface_data
                .surface_loader
                .get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface_data.surface,
                )
                .unwrap()
        };
        if fam.queue_count > 0 && present_supported {
            queue_family_indices.present_family = Some(index);
        }

        if queue_family_indices.graphics_family.is_some()
            && queue_family_indices.present_family.is_some()
        {
            break;
        }
        index += 1;
    }
    queue_family_indices
}
