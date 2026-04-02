use bytemuck;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Scene {
    pub win_size: [f32; 4],
    pub origin: [f32; 4],
    pub perspective: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Volume {
    pub cube_size: [f32; 3],
    pub _pad1: f32,
    pub block_size: [f32; 3],
    pub _pad2: f32,   // padding!
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RenderParams {
    pub cut_iso: [f32; 3],
    pub colormap: i32,
    pub diffuse_color: [f32; 4],
    pub bg_color: [f32; 3],
    pub transfer: i32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Interaction {
    pub zoom_min: [f32; 3],
    pub _pad1: f32,
    pub zoom_max: [f32; 3],
    pub _pad2: f32,   // padding!
    pub bbox_min: [f32; 3],
    pub _pad3: f32,   // padding!
    pub bbox_max: [f32; 3],
    pub _pad4: f32,   // padding!
}

