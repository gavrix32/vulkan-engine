use ash::ext::debug_utils;
use ash::{Instance, vk};
use raw_window_handle::RawDisplayHandle;
use std::ffi::{CStr, CString, c_char, c_void};

const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

pub struct VulkanState {
    _entry: ash::Entry,
    debug_utils_instance_messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    instance: Instance,
}

impl VulkanState {
    pub fn new(display_handle: RawDisplayHandle) -> Self {
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan library") };

        let mut debug_utils_messenger_create_info = Self::populate_debug_messenger_create_info();

        let instance = Self::create_instance(
            &entry,
            display_handle,
            &mut debug_utils_messenger_create_info,
        );
        let debug_utils_instance_messenger =
            Self::setup_debug_messenger(&entry, &instance, &debug_utils_messenger_create_info);

        Self {
            _entry: entry,
            debug_utils_instance_messenger,
            instance,
        }
    }

    fn create_instance(
        entry: &ash::Entry,
        display_handle: RawDisplayHandle,
        debug_create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT<'static>,
    ) -> Instance {
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Hello Triangle")
            .application_version(0)
            .engine_name(c"No Engine")
            .engine_version(0)
            .application_version(vk::make_api_version(0, 1, 0, 0));

        let required_extensions = Self::get_required_extensions(display_handle);

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&required_extensions);

        let layers_cstring: Vec<CString> = VALIDATION_LAYERS
            .iter()
            .map(|&s| CString::new(s).unwrap())
            .collect();
        let ptrs: Vec<*const c_char> = layers_cstring.iter().map(|s| s.as_ptr()).collect();

        if ENABLE_VALIDATION_LAYERS {
            if !Self::check_validation_layer_support(&entry) {
                panic!("validation layers requested, but not available!");
            }
            create_info = create_info
                .enabled_layer_names(&ptrs)
                .push_next(debug_create_info);
        }

        unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create VkInstance")
        }
    }

    fn check_validation_layer_support(entry: &ash::Entry) -> bool {
        let available_layers = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .expect("Failed to enumerate layer properties")
        };
        for layer_name in VALIDATION_LAYERS {
            let mut layer_found = false;

            for layer_properties in &available_layers {
                let name = unsafe { CStr::from_ptr(layer_properties.layer_name.as_ptr()) };
                if layer_name == name.to_str().unwrap() {
                    layer_found = true;
                    break;
                }
            }
            if !layer_found {
                return false;
            }
        }
        true
    }

    fn get_required_extensions(display_handle: RawDisplayHandle) -> Vec<*const c_char> {
        let mut extensions = ash_window::enumerate_required_extensions(display_handle)
            .unwrap()
            .to_vec();

        if ENABLE_VALIDATION_LAYERS {
            extensions.push(debug_utils::NAME.as_ptr());
        }
        extensions
    }

    fn setup_debug_messenger(
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

    fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
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
        SeverityFlag::VERBOSE => log::debug!("{:?} - {:?}", message_type, message),
        SeverityFlag::INFO => log::info!("{:?} - {:?}", message_type, message),
        SeverityFlag::WARNING => log::warn!("{:?} - {:?}", message_type, message),
        _ => log::error!("{:?} - {:?}", message_type, message),
    }
    vk::FALSE
}

impl Drop for VulkanState {
    fn drop(&mut self) {
        unsafe {
            if let Some((instance, messenger)) = self.debug_utils_instance_messenger.take() {
                instance.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None)
        };
    }
}
