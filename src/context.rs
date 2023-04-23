use ash::{vk, Device, Entry, Instance};

use crate::SurfaceData;

pub struct VkContext {
    pub _entry: Entry,
    pub instance: Instance,
    pub surface_data: SurfaceData,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
}

impl VkContext {
    pub fn new(
        entry: Entry,
        instance: Instance,
        surface_data: SurfaceData,
        physical_device: vk::PhysicalDevice,
        device: Device,
    ) -> VkContext {
        VkContext {
            _entry: entry,
            instance,
            surface_data,
            physical_device,
            device,
        }
    }

    pub fn get_mem_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface_data
                .surface_loader
                .destroy_surface(self.surface_data.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
