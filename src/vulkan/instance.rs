use crate::unsafe_vk_try;
use crate::vulkan::debug;
use ash::{ext, vk};
use raw_window_handle::RawDisplayHandle;
use std::ffi::{CStr, c_char};

pub struct Instance {
    pub entry: ash::Entry,
    pub ash_instance: ash::Instance,

    debug_instance: Option<ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl Instance {
    pub fn new(display_handle: RawDisplayHandle, validation: bool) -> Self {
        let entry = unsafe { ash::Entry::load() }.expect("Failed to load Vulkan library");

        let mut debug_utils_messenger_create_info = debug::create_debug_messenger_create_info();

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"No Engine")
            .engine_version(0)
            .api_version(vk::API_VERSION_1_3);

        let required_extensions = get_required_extensions(display_handle, validation);

        let mut debug_instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&required_extensions);

        let layer_cstring_pointers = debug::get_validation_layer_cstring_pointers();

        if validation {
            if !check_validation_layer_support(&entry) {
                panic!("Validation layers requested, but not available!");
            }
            debug_instance_create_info = debug_instance_create_info
                .enabled_layer_names(&layer_cstring_pointers.1)
                .push_next(&mut debug_utils_messenger_create_info);
        }

        let ash_instance = unsafe_vk_try!(entry.create_instance(&debug_instance_create_info, None));

        let (debug_instance, debug_messenger) = debug::setup_debug_messenger(
            &entry,
            &ash_instance,
            &debug_utils_messenger_create_info,
            validation,
        );

        Self {
            entry,
            ash_instance,
            debug_instance,
            debug_messenger,
        }
    }
}

fn get_required_extensions(
    display_handle: RawDisplayHandle,
    validation: bool,
) -> Vec<*const c_char> {
    let mut extensions = ash_window::enumerate_required_extensions(display_handle)
        .expect("Failed to enumerate extensions required for creating surface")
        .to_vec();

    if validation {
        extensions.push(ext::debug_utils::NAME.as_ptr());
    }
    extensions
}

fn check_validation_layer_support(entry: &ash::Entry) -> bool {
    let available_layers = unsafe_vk_try!(entry.enumerate_instance_layer_properties());
    for layer_name in debug::VALIDATION_LAYERS {
        let mut layer_found = false;

        for layer_properties in &available_layers {
            let name = unsafe { CStr::from_ptr(layer_properties.layer_name.as_ptr()) };
            if layer_name == name.to_str().expect("Failed to get a valid UTF-8 string") {
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

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            if let (Some(instance), Some(messenger)) =
                (self.debug_instance.take(), self.debug_messenger.take())
            {
                instance.destroy_debug_utils_messenger(messenger, None);
            }
            self.ash_instance.destroy_instance(None)
        }
    }
}
