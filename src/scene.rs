use crate::camera::Camera;
use crate::mesh::Mesh;

// TODO: Builder
pub struct Scene {
    pub camera: Camera,
    pub meshes: Vec<Mesh>,
}
