use crate::camera::Camera;
use crate::model::Model;

// TODO: Builder
// TODO: Point Lights, Directional Lights

#[derive(Default)]
pub struct Scene {
    pub camera: Camera,
    pub models: Vec<Model>,
}
