use wgpu::util::DeviceExt;
use wgpu::TextureView;
use crate::Mat4;
use crate::Texture;
use crate::VertexNDC;
use crate::Vec4;
pub(crate) struct VolumetricRenderer {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    texture_bind_group_layout: wgpu::BindGroupLayout,
    diffuse_bind_group: wgpu::BindGroup,
}

use std::collections::HashMap;
impl VolumetricRenderer {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, config: &wgpu::SurfaceConfiguration, buffers: &HashMap<&'static str, wgpu::Buffer>) -> Self {
        let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // rot matrix uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Mat4<f32>>() as _,
                        ),
                    },
                    count: None,
                },
                // window size uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Vec4<f32>>() as wgpu::BufferAddress,
                        ),
                    },
                    count: None,
                },
                // time uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Vec4<f32>>() as wgpu::BufferAddress,
                        ),
                    },
                    count: None,
                },
                // cam origin uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Vec4<f32>>() as wgpu::BufferAddress,
                        ),
                    },
                    count: None,
                },
                // cuts uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Vec4<f32>>() as wgpu::BufferAddress,
                        ),
                    },
                    count: None,
                },
                // perspective uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Vec4<f32>>() as wgpu::BufferAddress,
                        ),
                    },
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });

        let cube =
            Texture::from_raw_bytes::<f32>(&device, &queue, None, (1, 1, 1), 4, "cube").unwrap();

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cube.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&cube.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["rotmat"],
                        offset: 0,
                        size: wgpu::BufferSize::new(
                            std::mem::size_of::<Mat4<f32>>() as wgpu::BufferAddress
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["window_size"],
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["time"],
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["cam_origin"],
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["cuts"],
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["perspective"],
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        // uniform buffer
        let vs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cube vert shader"),
            source: wgpu::ShaderSource::Glsl {
                #[cfg(not(target_arch = "wasm32"))]
                shader: std::str::from_utf8(&std::fs::read("src/shaders/cube.vert").unwrap())
                    .unwrap()
                    .into(),
                #[cfg(target_arch = "wasm32")]
                shader: include_str!("shaders/cube.vert").into(),
                stage: wgpu::naga::ShaderStage::Vertex,
                defines: Default::default(),
            },
        });
        let fs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cube frag shader"),
            source: wgpu::ShaderSource::Glsl {
                #[cfg(not(target_arch = "wasm32"))]
                shader: std::str::from_utf8(&std::fs::read("src/shaders/cube.frag").unwrap())
                    .unwrap()
                    .into(),
                #[cfg(target_arch = "wasm32")]
                shader: include_str!("shaders/cube.frag").into(),
                stage: wgpu::naga::ShaderStage::Fragment,
                defines: Default::default(),
            },
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                //immediate_size: 0,
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                buffers: &[VertexNDC::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            //multiview_mask: None,
            multiview: None,
            cache: None, // 6.
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&[
                VertexNDC { ndc: [-1.0, -1.0] },
                VertexNDC { ndc: [1.0, -1.0] },
                VertexNDC { ndc: [1.0, 1.0] },
                VertexNDC { ndc: [-1.0, 1.0] },
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&[0, 1, 2, 0, 2, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            render_pipeline,
            vertex_buffer,
            index_buffer,
            diffuse_bind_group,
            texture_bind_group_layout,
        }
    }

    pub(crate) fn set_volume(&mut self, device: &wgpu::Device, buffers: &HashMap<&'static str, wgpu::Buffer>, volume: Texture) {
        self.diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&volume.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&volume.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["rotmat"],
                        offset: 0,
                        size: wgpu::BufferSize::new(
                            std::mem::size_of::<Mat4<f32>>() as wgpu::BufferAddress
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["window_size"],
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["time"],
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["cam_origin"],
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["cuts"],
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers["perspective"],
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
            ],
            label: Some("diffuse_bind_group"),
        });
    }

    pub(crate) fn render_frame(&self, encoder: &mut wgpu::CommandEncoder, window_surface_view: &TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &window_surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.01,
                        g: 0.01,
                        b: 0.01,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            //multiview_mask: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass
            .set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..6, 0, 0..1);
    }
}