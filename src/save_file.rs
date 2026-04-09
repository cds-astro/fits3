use std::num::NonZeroU32;
use std::io::Cursor;
use image::ImageBuffer;
use image::Rgba;

#[cfg(target_arch = "wasm32")]
pub async fn save_texture_to_png(
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture: wgpu::Texture,
    config: wgpu::SurfaceConfiguration,
    vw: u32,
    vh: u32,
    vx: u32,
    vy: u32,
    filename: &str,
) {
    let width = config.width;
    let height = config.height;

    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row =
        ((unpadded_bytes_per_row + align - 1) / align) * align;

    let buffer_size = (padded_bytes_per_row * height) as wgpu::BufferAddress;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy texture → buffer
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("texture copy encoder"),
        });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(
                    padded_bytes_per_row
                ),
                rows_per_image: Some(
                    height
                ),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    // Await GPU
    use futures_intrusive::channel::shared::oneshot_channel;

    let buffer_slice = buffer.slice(..);

    let (sender, receiver) = oneshot_channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    receiver.receive().await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();

    // Remove padding
    let mut pixels = vec![0u8; (width * height * 4) as usize];

    for y in 0..height as usize {
        let src_offset = y * padded_bytes_per_row as usize;
        let dst_offset = y * (width * 4) as usize;

        pixels[dst_offset..dst_offset + (width * 4) as usize]
            .copy_from_slice(
                &data[src_offset..src_offset + (width * 4) as usize],
            );
    }

    drop(data);
    buffer.unmap();

    let mut png_bytes: Vec<u8> = Vec::new();


    {
        let img: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(width, height, pixels.to_vec())
                .expect("Invalid buffer size");

        let img = image::imageops::resize(&img, vw, vh, image::imageops::FilterType::Nearest);

        let mut cursor = Cursor::new(&mut png_bytes);

        img.write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
    }

    // --- SAVE ---
    use wasm_bindgen::JsCast;
    use web_sys::{Blob, Url};

    let array = js_sys::Uint8Array::from(png_bytes.as_slice());

    let blob_parts = js_sys::Array::new();
    blob_parts.push(&array.buffer());

    let blob = Blob::new_with_u8_array_sequence(&blob_parts)
        .unwrap();

    let url = Url::create_object_url_with_blob(&blob).unwrap();

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();

    let a = document.create_element("a").unwrap();
    a.set_attribute("href", &url).unwrap();
    a.set_attribute("download", filename).unwrap();

    let body = document.body().unwrap();
    body.append_child(&a).unwrap();

    let a: web_sys::HtmlElement = a.dyn_into().unwrap();
    a.click();

    body.remove_child(&a).unwrap();
    Url::revoke_object_url(&url).unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
pub fn save_texture_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    config: &wgpu::SurfaceConfiguration,
    vw: u32,
    vh: u32,
    vx: u32,
    vy: u32,
    filename: &str,
) {
    let width = config.width;
    let height = config.height;

    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row =
        ((unpadded_bytes_per_row + align - 1) / align) * align;

    let buffer_size = (padded_bytes_per_row * height) as wgpu::BufferAddress;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy texture → buffer
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("texture copy encoder"),
        });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(
                    padded_bytes_per_row
                ),
                rows_per_image: Some(
                    height
                ),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    // Await GPU
    let buffer_slice = buffer.slice(..);

    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let data = buffer_slice.get_mapped_range();

    // Remove padding
    let mut pixels = vec![0u8; (width * height * 4) as usize];

    for y in 0..height as usize {
        let src_offset = y * padded_bytes_per_row as usize;
        let dst_offset = y * (width * 4) as usize;

        pixels[dst_offset..dst_offset + (width * 4) as usize]
            .copy_from_slice(
                &data[src_offset..src_offset + (width * 4) as usize],
            );

        for pixel in pixels[dst_offset..dst_offset + (width * 4) as usize].chunks_exact_mut(4) {
            pixel.swap(0, 2); // swap R <-> B
        }
    }

    drop(data);
    buffer.unmap();

    // --- SAVE ---
    let img: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(width, height, pixels)
            .expect("Failed to create image buffer");

    let img = image::imageops::resize(&img, vw, vh, image::imageops::FilterType::Nearest);

    img.save(filename).expect("Failed to save PNG");
    println!("Saved PNG to {}", filename);
}