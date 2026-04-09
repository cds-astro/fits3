use fitsrs::card::Value;
use std::ops::Range;

use crate::Texture;
use fitsrs::HDU;
use std::io::Cursor;

use glam::BVec3;
use glam::UVec3;

use std::convert::TryInto;

use fitsrs::Fits;

use glam::Vec3;

pub(crate) struct Cube {
    pub data: Vec<f32>,
    pub size: (u32, u32, u32),
    pub b_size: (u32, u32, u32),
    pub mincut: f32,
    pub maxcut: f32,
    pub texture: Texture,
    pub downsampled_texture: Texture,

    #[cfg(target_arch = "wasm32")]
    pub wcs: fitsrs::WCS,
}

impl Cube {
    pub(crate) fn from_fits<R>(
        reader: Cursor<R>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Self, &'static str>
    where
        R: AsRef<[u8]> + std::fmt::Debug,
    {
        let mut fits = Fits::from_reader(reader);

        if let Some(Ok(hdu)) = fits.next() {
            match hdu {
                HDU::Primary(hdu) => {
                    let header = hdu.get_header();

                    if let (
                        Some(Value::Integer { value: w, .. }),
                        Some(Value::Integer { value: h, .. }),
                        Some(Value::Integer { value: d, .. }),
                        Some(Value::Integer { value: b, .. }),
                    ) = (
                        header.get("NAXIS1"),
                        header.get("NAXIS2"),
                        header.get("NAXIS3"),
                        header.get("BITPIX"),
                    ) {
                        let image = fits.get_data(&hdu);

                        let d1 = *w as u32;
                        let d2 = *h as u32;
                        let mut d3 = *d as u32;

                        if d3 == 1 {
                            // parse NAXIS4 instead it there is
                            if let Some(Value::Integer { value, .. }) = header.get("NAXIS4") {
                                d3 = *value as u32;
                            }
                        }

                        let raw_bytes = image.raw_bytes();

                        let (data, cuts) = match b {
                            -32 => {
                                let floats: Vec<f32> = raw_bytes
                                    .chunks_exact(4)
                                    .map(|b| f32::from_be_bytes(b.try_into().unwrap()))
                                    .collect();

                                //let minmax = first_and_last_percent_f32(&mut floats, 0.01, 99.5);
                                let cuts = estimate_default_cuts_from_variance(&floats);
                                (floats, cuts)
                            }
                            8 => {
                                todo!();
                                /*let mut bytes: Vec<u8> = data.to_vec();
                                let range = first_and_last_percent(&mut bytes, 0.01, 99.5);
                                (range.start as f32)..(range.end as f32)*/
                            }
                            16 => {
                                todo!();
                                /*let mut shorts: Vec<i16> = data
                                    .chunks_exact(2)
                                    .map(|b| i16::from_be_bytes(b.try_into().unwrap()))
                                    .collect();

                                let range = first_and_last_percent(&mut shorts, 0.01, 99.5);
                                (range.start as f32)..(range.end as f32)*/
                            }
                            32 => {
                                /*let mut int32: Vec<i32> = data
                                    .chunks_exact(4)
                                    .map(|b| i32::from_be_bytes(b.try_into().unwrap()))
                                    .collect();

                                let range = first_and_last_percent(&mut int32, 0.01, 99.5);
                                (range.start as f32)..(range.end as f32)*/
                                todo!();
                            }
                            64 => {
                                /*let mut int64: Vec<i64> = data
                                    .chunks_exact(8)
                                    .map(|b| i64::from_be_bytes(b.try_into().unwrap()))
                                    .collect();

                                let range = first_and_last_percent(&mut int64, 0.01, 99.5);
                                (range.start as f32)..(range.end as f32)*/
                                todo!();
                            }
                            _ => {
                                return Err("F32 only supported");
                            }
                        };

                        let bz = (d3 / 64).clamp(16, 128);
                        let bx = 32_u32;
                        let by = 32_u32;

                        let downsampled_raw_bytes = downsample_8x(
                            &data,
                            d1 as usize,
                            d2 as usize,
                            d3 as usize,
                            bx as usize,
                            by as usize,
                            bz as usize,
                        )
                        .iter()
                        .flat_map(|p| p.to_le_bytes())
                        .collect::<Vec<u8>>();

                        let original_dim = (d1, d2, d3);
                        let padding = (
                            (bx - (d1 % bx)) % bx,
                            (by - (d2 % by)) % by,
                            (bz - (d3 % bz)) % bz,
                        );

                        let texture = Texture::from_raw_bytes::<f32>(
                            device,
                            queue,
                            Some(raw_bytes),
                            original_dim,
                            padding,
                            4,
                            "cube",
                        )?;

                        let downsampled_texture = Texture::from_raw_bytes::<f32>(
                            device,
                            queue,
                            Some(&downsampled_raw_bytes),
                            (d1.div_ceil(bx), d2.div_ceil(by), d3.div_ceil(bz)),
                            (0, 0, 0),
                            4,
                            "downgraded_cube",
                        )?;

                        let size = (original_dim.0, original_dim.1, original_dim.2);

                        #[cfg(target_arch = "wasm32")]
                        let wcs = hdu.wcs().map_err(|_| "wcs not found")?;

                        let b_size = (bx, by, bz);
                        // Build the downgrade resolued cube for faster raytracing.
                        // This cube will be first sampled to know whether it is interesting
                        // to sample the full resolued one or to skip to the next big voxel (8x8x8)
                        Ok(Self {
                            data,
                            size,
                            b_size,
                            mincut: cuts.start,
                            maxcut: cuts.end,
                            #[cfg(target_arch = "wasm32")]
                            wcs,
                            texture,
                            downsampled_texture,
                        })
                    } else {
                        Err("FITS image extension not found")
                    }
                }
                _ => Err("FITS image extension not found"),
            }
        } else {
            Err("Is not a FITS file")
        }
    }

    pub fn extract_spectra_from_origin_and_dir(
        &self,
        origin: &Vec3,
        dir: &Vec3,
        zoom_box: Option<(Vec3, Vec3)>,
    ) -> Option<Vec<f32>> {
        let p_cam = origin;

        // vector director from the cam origin to the pixel on screen
        // traditional perspective director vector
        // orthographic perspective
        let r = dir.normalize();

        let bbox_min = Vec3::new(-0.5, -0.5, -0.5);
        let bbox_max = Vec3::new(0.5, 0.5, 0.5);

        let t_low = (bbox_min - p_cam) / r;
        let t_high = (bbox_max - p_cam) / r;

        let t_close = t_low.min(t_high);
        let t_far = t_low.max(t_high);

        let t_c = t_close.max_element();
        let t_f = t_far.min_element();

        if t_f > t_c {
            let abs_r = r.abs();
            let inv_r = 1.0 / r;
            let cube_size = UVec3::from(self.size).as_vec3();

            let (zl, zh) = zoom_box.unwrap_or((Vec3::ZERO, Vec3::ONE));

            let z_dscale = zh - zl;
            let z_scale = z_dscale;
            let z_offset = zl;

            let inv_dir = abs_r * cube_size * z_scale;
            let step = 1.0 / inv_dir.max_element();

            let size = cube_size;
            let size_inv = 1.0 / size;

            let _dr = r * step;
            // absolute sampling point
            // scaled to the origin of the cube
            let mut t = t_c;
            // p in [0; 1]
            let mut p = p_cam + r * t + Vec3::splat(0.5);

            let step_dir = r.signum();
            let mut cell = (p * size).floor();
            let next_boundary = (cell + step_dir.max(Vec3::ZERO)) * size_inv;

            let mut t_max = t + (next_boundary - p) * inv_r;
            let mut t_delta = size_inv * inv_r.abs();

            let eps = 1e-8;
            let zero_dir: BVec3 = r.abs().cmplt(Vec3::splat(eps));
            let big = Vec3::splat(1e30);

            t_delta = Vec3::select(zero_dir, big, t_delta);
            t_max = Vec3::select(zero_dir, big, t_max);

            let uv_step = size_inv * step_dir;
            let mut uv = z_offset + (cell + 0.5) * size_inv * z_dscale;

            let mut values = vec![];

            // DDA-style voxel traversing
            while t < t_f {
                values.push(self.probe(&uv).unwrap());

                let t_next = t_max.min_element();

                let t_prev = t;
                let b_mask = t_max.cmple(Vec3::splat(t_next));
                let mask = Vec3::select(b_mask, Vec3::ONE, Vec3::ZERO);

                t = t_next;

                cell += mask * step_dir;
                t_max += mask * t_delta;

                p += r * (t - t_prev);

                uv += mask * uv_step * z_dscale;
            }

            Some(values)
        } else {
            None
        }
    }

    fn probe(&self, uv: &Vec3) -> Option<f32> {
        let uv = uv.clamp(Vec3::ZERO, Vec3::ONE - Vec3::splat(1e-6));

        let xyz = Vec3::new(
            uv.x * self.size.0 as f32,
            uv.y * self.size.1 as f32,
            uv.z * self.size.2 as f32,
        )
        .as_uvec3();

        let index = (xyz.x + xyz.y * self.size.0 + xyz.z * self.size.0 * self.size.1) as usize;

        self.data.get(index).copied()
    }
}

fn downsample_8x(
    input: &[f32],
    size_x: usize,
    size_y: usize,
    size_z: usize,
    bx: usize,
    by: usize,
    bz: usize,
) -> Vec<f32> {
    let new_x = size_x.div_ceil(bx);
    let new_y = size_y.div_ceil(by);
    let new_z = size_z.div_ceil(bz);

    let mut output = vec![0.0; new_x * new_y * new_z];

    for oz in 0..new_z {
        for oy in 0..new_y {
            for ox in 0..new_x {
                let start_x = ox * bx;
                let start_y = oy * by;
                let start_z = oz * bz;

                let end_x = (start_x + bx).min(size_x);
                let end_y = (start_y + by).min(size_y);
                let end_z = (start_z + bz).min(size_z);

                let mut max = f32::NEG_INFINITY;

                let mut is_nan = true;
                for z in start_z..end_z {
                    for y in start_y..end_y {
                        for x in start_x..end_x {
                            let idx = x + size_x * (y + size_y * z);
                            let p = input[idx];
                            if !p.is_nan() {
                                is_nan = false;
                                max = p.max(max);
                            }
                        }
                    }
                }

                if is_nan {
                    max = f32::NAN;
                }

                let out_idx = ox + new_x * (oy + new_y * oz);
                output[out_idx] = max;
            }
        }
    }

    output
}

pub fn first_and_last_percent_f32(
    slice: &mut [f32],
    mut first_percent: f32,
    mut last_percent: f32,
) -> Range<f32> {
    if slice.is_empty() {
        return 0.0..0.0;
    }

    if first_percent > last_percent {
        std::mem::swap(&mut first_percent, &mut last_percent);
    }

    // Move all NaNs to the end
    let valid_len = {
        let mut i = 0;
        for j in 0..slice.len() {
            if !slice[j].is_nan() {
                slice.swap(i, j);
                i += 1;
            }
        }
        i
    };

    if valid_len == 0 {
        return f32::NAN..f32::NAN;
    }

    let valid = &mut slice[..valid_len];

    let i1 = (first_percent.clamp(0.0, 100.0) as usize * valid_len) / 100;
    let i2 = (last_percent.clamp(0.0, 100.0) as usize * valid_len) / 100;

    let min_val = {
        let (_, min_val, _) = valid.select_nth_unstable_by(i1, |a, b| a.total_cmp(b));
        *min_val
    };
    let max_val = {
        let (_, max_val, _) = valid.select_nth_unstable_by(i2, |a, b| a.total_cmp(b));
        *max_val
    };

    min_val..max_val
}

pub fn estimate_default_cuts_from_variance(slice: &[f32]) -> Range<f32> {
    if slice.is_empty() {
        return 0.0..0.0;
    }

    let valid_len = slice.len();

    if valid_len == 0 {
        return f32::NAN..f32::NAN;
    }

    //let valid = &mut slice[..valid_len];

    /*let (_, median, _) = valid.select_nth_unstable_by(valid_len / 2, |a, b| {
        a.total_cmp(&b)
    });
    let median = *median;*/

    let num_samples = 50000.min(valid_len);
    let s = (valid_len / num_samples).max(1);

    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let mut n = 0;

    for i in (0..slice.len()).step_by(s) {
        let v = slice[i];
        if !v.is_nan() {
            sum += v;
            sum2 += v * v;
            n += 1;
        }
    }

    let mean = sum / (n as f32);
    let sigma = ((sum2 / (n as f32)) - mean * mean).sqrt();

    (sigma)..(15.0 * sigma)
}

/*
pub fn first_and_last_percent<T>(
    slice: &mut [T],
    mut first_percent: f32,
    mut last_percent: f32,
) -> Range<T>
where
    T: std::cmp::Ord + cgmath::Zero + Copy
{
    if slice.is_empty() {
        return T::zero()..T::zero();
    }

    if first_percent > last_percent {
        std::mem::swap(&mut first_percent, &mut last_percent);
    }


    let n = slice.len();
    let i1 = (first_percent.clamp(0.0, 100.0) as usize * n) / 100;
    let i2 = (last_percent.clamp(0.0, 100.0) as usize * n) / 100;

    let (_, min_val, _) =
        slice.select_nth_unstable(i1);
    let min_val = *min_val;
    let (_, max_val, _) =
        slice.select_nth_unstable(i2);
    let max_val = *max_val;

    min_val..max_val
}
*/
