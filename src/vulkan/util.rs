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

pub fn rgb_to_rgba(rgb_data: &[u8]) -> Vec<u8> {
    rgb_data
        .chunks_exact(3)
        .flat_map(|rgb_pixel| [rgb_pixel[0], rgb_pixel[1], rgb_pixel[2], 255])
        .collect()
}

pub fn r_to_rgba(rgb_data: &[u8]) -> Vec<u8> {
    rgb_data
        .chunks_exact(1)
        .flat_map(|rgb_pixel| [rgb_pixel[0], 0, 0, 255])
        .collect()
}

pub fn rg_to_rgba(rgb_data: &[u8]) -> Vec<u8> {
    rgb_data
        .chunks_exact(2)
        .flat_map(|rgb_pixel| [rgb_pixel[0], rgb_pixel[1], 0, 255])
        .collect()
}
