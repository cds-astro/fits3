use glam::DVec3;

pub(crate) fn lonlat2xyz(lon: f64, lat: f64) -> DVec3 {
    let lat_s = lat.sin();
    let lat_c = lat.cos();
    let lon_s = lon.sin();
    let lon_c = lon.cos();

    DVec3::new(lat_c * lon_s, lat_s, lat_c * lon_c)
}
