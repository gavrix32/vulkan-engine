use std::ffi::{c_void, CStr};
use ash::{vk, Instance};
use ash::ext::debug_utils;

pub const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
pub const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

pub fn setup_debug_messenger(
    entry: &ash::Entry,
    instance: &Instance,
    create_info: &vk::DebugUtilsMessengerCreateInfoEXT,
) -> Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
    if !ENABLE_VALIDATION_LAYERS {
        return None;
    }

    let debug_utils_instance = debug_utils::Instance::new(entry, instance);
    let debug_utils_messenger = unsafe {
        debug_utils_instance
            .create_debug_utils_messenger(&create_info, None)
            .expect("Failed to create debug utils messenger")
    };
    Some((debug_utils_instance, debug_utils_messenger))
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
    match message_severity {
        SeverityFlag::VERBOSE => log::trace!("{:?} - {:?}", message_type, message),
        SeverityFlag::INFO => log::info!("{:?} - {:?}", message_type, message),
        SeverityFlag::WARNING => log::warn!("{:?} - {:?}", message_type, message),
        _ => log::error!("{:?} - {:?}", message_type, message),
    }
    vk::FALSE
}