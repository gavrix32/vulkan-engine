use glam::{EulerRot, Mat4, Quat, Vec3};

#[derive(Default, Clone, Copy)]
pub struct Camera {
    pub pos: Vec3,
    pub quat: Quat,
}

impl Camera {
    pub fn view(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.quat, self.pos).inverse()
    }

    pub fn move_local(&mut self, x: f32, y: f32, z: f32) {
        let right = self.quat * Vec3::X;
        let up = self.quat * Vec3::Y;
        let forward = self.quat * Vec3::NEG_Z;

        self.pos += right * x + up * y + forward * z;
    }

    pub fn move_local_x(&mut self, speed: f32) {
        self.move_local(speed, 0.0, 0.0);
    }

    pub fn move_local_y(&mut self, speed: f32) {
        self.move_local(0.0, speed, 0.0);
    }

    pub fn move_local_z(&mut self, speed: f32) {
        self.move_local(0.0, 0.0, speed);
    }

    pub fn euler_rot(&self) -> (f32, f32, f32) {
        self.quat.to_euler(EulerRot::YXZ)
    }

    pub fn set_euler_rot(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.quat = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }
}
