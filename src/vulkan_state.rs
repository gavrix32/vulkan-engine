use ash::ext::debug_utils;
use ash::{Instance, vk};
use raw_window_handle::RawDisplayHandle;
use std::ffi::{CStr, CString, c_char};
use crate::debug;

pub struct VulkanState {
    _entry: ash::Entry,
    debug_utils_instance_messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    instance: Instance,
}

impl VulkanState {
    pub fn new(display_handle: RawDisplayHandle) -> Self {
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan library") };

        let mut debug_utils_messenger_create_info = debug::create_debug_messenger_create_info();

        let instance = Self::create_instance(
            &entry,
            display_handle,
            &mut debug_utils_messenger_create_info,
        );
        let debug_utils_instance_messenger =
            debug::setup_debug_messenger(&entry, &instance, &debug_utils_messenger_create_info);

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

        let layers_cstring: Vec<CString> = debug::VALIDATION_LAYERS
            .iter()
            .map(|&s| CString::new(s).unwrap())
            .collect();
        let ptrs: Vec<*const c_char> = layers_cstring.iter().map(|s| s.as_ptr()).collect();

        if debug::ENABLE_VALIDATION_LAYERS {
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
        for layer_name in debug::VALIDATION_LAYERS {
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

        if debug::ENABLE_VALIDATION_LAYERS {
            extensions.push(debug_utils::NAME.as_ptr());
        }
        extensions
    }
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
