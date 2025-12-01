use crate::camera::Camera;
use crate::mesh::Mesh;

// TODO: Builder
// TODO: Point Lights, Directional Lights
pub struct Scene {
    pub camera: Camera,
    pub meshes: Vec<Mesh>,
}
