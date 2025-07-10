use ash::ext;
use ash::vk;
use std::ffi::{CStr, CString, c_char, c_void};

pub const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
pub const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

pub fn setup_debug_messenger(
    entry: &ash::Entry,
    instance: &ash::Instance,
    create_info: &vk::DebugUtilsMessengerCreateInfoEXT,
) -> (Option<ext::debug_utils::Instance>, Option<vk::DebugUtilsMessengerEXT>) {
    if !ENABLE_VALIDATION_LAYERS {
        return (None, None);
    }

    let debug_utils_instance = ext::debug_utils::Instance::new(entry, instance);
    let debug_utils_messenger = unsafe {
        debug_utils_instance
            .create_debug_utils_messenger(&create_info, None)
            .expect("Failed to create debug utils messenger")
    };
    
    (Some(debug_utils_instance), Some(debug_utils_messenger))
}

pub fn create_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
}

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as SeverityFlag;

    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    let message_str = message.to_str().unwrap();

    match message_severity {
        SeverityFlag::VERBOSE => log::trace!("{:?} - {}", message_type, message_str),
        SeverityFlag::INFO => log::info!("{:?} - {}", message_type, message_str),
        SeverityFlag::WARNING => log::warn!("{:?} - {}", message_type, message_str),
        _ => log::error!("{:?} - {}", message_type, message_str),
    }
    vk::FALSE
}

pub fn get_validation_layer_cstring_pointers() -> (Vec<CString>, Vec<*const c_char>) {
    let layers_cstring: Vec<CString> = VALIDATION_LAYERS
        .iter()
        .map(|&s| CString::new(s).unwrap())
        .collect();
    let pointers: Vec<*const c_char> = layers_cstring.iter().map(|s| s.as_ptr()).collect();
    (layers_cstring, pointers)
}
