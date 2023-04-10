use std::os::raw::c_char;

/// Helper function to convert [c_char; SIZE] to string
pub fn vk_to_string(raw_string_array: &[c_char]) -> String {
    // Implementation 2
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        std::ffi::CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}
