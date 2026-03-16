pub(crate) fn compute_moment0(cube: &crate::Cube) -> Vec<u8> {
    let naxis = &cube.dim;

    let pixels_per_slice = naxis.0 * naxis.1;
    let num_voxels = (naxis.0 * naxis.1 * naxis.2) as usize;
    let mut rgba = Vec::with_capacity(num_voxels * std::mem::size_of::<f32>());

    let cut_factor = 256.0 / (cube.maxcut - cube.mincut);

    for y in (0..naxis.1).rev() {
        for x in 0..naxis.0 {
            let mut n = 0;
            let mut sum = 0.0;

            let mut i = (x + y*naxis.0) as usize;
            for z in 0..naxis.2 {
                if !cube.data[i].is_nan() {
                    sum += cube.data[i];
                    n += 1;
                }

                i = i + pixels_per_slice as usize;
            }

            let v = sum / (n as f32);
            let p = ((v - cube.mincut) * cut_factor) as u8;
            rgba.extend([p, p, p, 255])
        }
    }

    rgba
}