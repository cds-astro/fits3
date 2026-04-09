use crate::Cube;

pub(crate) fn compute_moment0(cube: &Cube) -> Vec<u8> {
    let naxis = &cube.size;

    let pixels_per_slice = naxis.0 * naxis.1;
    let num_voxels = (naxis.0 * naxis.1 * naxis.2) as usize;
    let mut rgba = Vec::with_capacity(num_voxels * std::mem::size_of::<f32>());

    let max_val = cube
        .data
        .iter()
        .filter(|x| !x.is_nan())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&1.0);
    let min_val = cube
        .data
        .iter()
        .filter(|x| !x.is_nan())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);

    let cut_factor = 256.0 / (max_val - min_val);

    for y in (0..naxis.1).rev() {
        for x in 0..naxis.0 {
            let mut n = 0;
            let mut sum = 0.0;

            let mut i = (x + y * naxis.0) as usize;
            for _ in 0..naxis.2 {
                if !cube.data[i].is_nan() {
                    sum += cube.data[i];
                    n += 1;
                }

                i += pixels_per_slice as usize;
            }

            if n == 0 {
                rgba.extend([255, 0, 0, 255])
            } else {
                let v = sum / (n as f32);
                let p = ((v - min_val) * cut_factor) as u8;
                rgba.extend([p, p, p, 255])
            }
        }
    }

    rgba
}
