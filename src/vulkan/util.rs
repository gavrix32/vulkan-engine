#[macro_export]
macro_rules! vk_try {
    ($expr:expr) => {{
        let result = $expr;
        match result {
            Ok(value) => value,
            Err(err) => {
                let msg = format!("Vulkan call failed: `{}` -> {:?}", stringify!($expr), err);
                log::error!("{}", msg);
                panic!("{}", msg)
            }
        }
    }};
}

#[macro_export]
macro_rules! unsafe_vk_try {
    ($expr:expr) => {{
        let result = unsafe { $expr };
        match result {
            Ok(value) => value,
            Err(err) => {
                let msg = format!("Vulkan call failed: `{}` -> {:?}", stringify!($expr), err);
                log::error!("{}", msg);
                panic!("{}", msg)
            }
        }
    }};
}