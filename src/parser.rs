use crate::mesh::Primitive;
use crate::vertex::Vertex;
use glam::Mat4;
use gltf::{Node, buffer};
use log::info;
use mikktspace::Geometry;

pub(crate) struct MeshView<'a> {
    pub(crate) vertices: &'a mut Vec<Vertex>,
    pub(crate) indices: &'a Vec<u32>,
}

impl<'a> Geometry for MeshView<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        let idx = self.indices[face * 3 + vert] as usize;
        self.vertices[idx].position
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let idx = self.indices[face * 3 + vert] as usize;
        self.vertices[idx].normal
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let idx = self.indices[face * 3 + vert] as usize;
        self.vertices[idx].tex_coord
    }

    fn set_tangent(
        &mut self,
        tangent: [f32; 3],
        _bi_tangent: [f32; 3],
        _f_mag_s: f32,
        _f_mag_t: f32,
        bi_tangent_preserves_orientation: bool,
        face: usize,
        vert: usize,
    ) {
        let sign = if bi_tangent_preserves_orientation {
            1.0
        } else {
            -1.0
        };
        let idx = self.indices[face * 3 + vert] as usize;
        self.vertices[idx].tangent = [tangent[0], tangent[1], tangent[2], sign];
    }
}

fn traverse_node(
    node: Node,
    buffers: &Vec<buffer::Data>,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    primitives: &mut Vec<Primitive>,
    parent_transform: Mat4,
    default_idx: usize,
) {
    info!("Node: {}", node.name().unwrap_or("Unnamed"));

    let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let model_matrix = parent_transform * local_transform;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let material = primitive.material();

            let albedo_index = material
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| info.texture().source().index());

            let normal_index = material
                .normal_texture()
                .map(|info| info.texture().source().index());

            let metallic_roughness_index = material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .map(|info| info.texture().source().index());

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let tex_coords = reader
                .read_tex_coords(0)
                .map(|tc| tc.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]]);

            let first_index = indices.len() as u32;

            let indices_iter = reader
                .read_indices()
                .expect("Primitive indices not found")
                .into_u32();
            let index_count = indices_iter.len() as u32;

            for index in indices_iter {
                indices.push(vertices.len() as u32 + index);
            }

            for (i, (pos, norm)) in reader
                .read_positions()
                .expect("Primitive positions not found")
                .zip(reader.read_normals().expect("Primitive normals not found"))
                .enumerate()
            {
                vertices.push(Vertex {
                    position: [pos[0], pos[1], pos[2]],
                    normal: [norm[0], norm[1], norm[2]],
                    tangent: [0.0; 4],
                    tex_coord: tex_coords.get(i).copied().unwrap_or([0.0, 0.0]),
                    material_indices: [
                        albedo_index.unwrap_or(default_idx) as u32,
                        normal_index.unwrap_or(default_idx) as u32,
                        metallic_roughness_index.unwrap_or(default_idx) as u32,
                    ],
                });
            }

            let primitive_info = Primitive {
                first_index,
                index_count,
                model_matrix,
            };
            primitives.push(primitive_info);
        }
    }

    for child in node.children() {
        traverse_node(
            child,
            buffers,
            vertices,
            indices,
            primitives,
            model_matrix,
            default_idx,
        );
    }
}

pub(crate) fn parse_model(
    document: &gltf::Document,
    buffers: &Vec<buffer::Data>,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    primitives: &mut Vec<Primitive>,
    default_idx: usize,
) {
    for scene in document.scenes() {
        info!("Scene: {}", scene.name().unwrap_or("Unnamed"));
        for node in scene.nodes() {
            traverse_node(
                node,
                &buffers,
                vertices,
                indices,
                primitives,
                Mat4::IDENTITY,
                default_idx,
            );
        }
    }
}
