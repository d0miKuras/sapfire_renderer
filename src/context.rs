use ash::{vk, Device, Entry, Instance};

use crate::SurfaceData;

pub struct VkContext {
    pub entry: Entry,
    pub instance: Instance,
    pub surface_data: SurfaceData,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl VkContext {
    pub fn new(
        entry: Entry,
        instance: Instance,
        surface_data: SurfaceData,
        physical_device: vk::PhysicalDevice,
        device: Device,
    ) -> VkContext {
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        VkContext {
            entry,
            instance,
            surface_data,
            physical_device,
            device,
            physical_device_memory_properties,
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
