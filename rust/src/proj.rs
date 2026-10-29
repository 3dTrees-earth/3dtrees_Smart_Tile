//! PROJ through its C API, loaded at run time.
//!
//! Python SmartTile uses pyproj, which wraps the same libproj; calling it
//! directly keeps coordinate transforms bit-identical without build-time
//! headers. The library is found via SMARTTILE_LIBPROJ, the conda prefix of
//! the image, or the loader path.

use std::ffi::{c_char, c_double, c_int, c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::OnceLock;

use libloading::Library;

#[repr(C)]
#[derive(Clone, Copy)]
struct Coord([c_double; 4]);

struct Api {
    _lib: Library,
    context_create: unsafe extern "C" fn() -> *mut c_void,
    context_set_search_paths: unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char),
    crs_to_crs: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, *mut c_void) -> *mut c_void,
    normalize: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void,
    trans: unsafe extern "C" fn(*mut c_void, c_int, Coord) -> Coord,
    destroy: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    errno: unsafe extern "C" fn(*const c_void) -> c_int,
    errno_string: unsafe extern "C" fn(*mut c_void, c_int) -> *const c_char,
    create: unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void,
    as_wkt: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, *const *const c_char) -> *const c_char,
    get_type: unsafe extern "C" fn(*const c_void) -> c_int,
    get_name: unsafe extern "C" fn(*const c_void) -> *const c_char,
    source_crs: unsafe extern "C" fn(*mut c_void, *const c_void) -> *mut c_void,
    sub_crs: unsafe extern "C" fn(*mut c_void, *const c_void, c_int) -> *mut c_void,
    coordinate_system: unsafe extern "C" fn(*mut c_void, *const c_void) -> *mut c_void,
    axis_count: unsafe extern "C" fn(*mut c_void, *const c_void) -> c_int,
    axis_info: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, *mut *const c_char, *mut *const c_char,
                                    *mut *const c_char, *mut c_double, *mut *const c_char, *mut *const c_char,
                                    *mut *const c_char) -> c_int,
    identify: unsafe extern "C" fn(*mut c_void, *const c_void, *const c_char, *const *const c_char, *mut *mut c_int) -> *mut c_void,
    list_count: unsafe extern "C" fn(*const c_void) -> c_int,
    list_get: unsafe extern "C" fn(*mut c_void, *const c_void, c_int) -> *mut c_void,
    list_destroy: unsafe extern "C" fn(*mut c_void),
    int_list_destroy: unsafe extern "C" fn(*mut c_int),
    id_auth_name: unsafe extern "C" fn(*const c_void, c_int) -> *const c_char,
    id_code: unsafe extern "C" fn(*const c_void, c_int) -> *const c_char,
    is_equivalent: unsafe extern "C" fn(*const c_void, *const c_void, c_int) -> c_int,
}

fn prefixes() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for var in ["CONDA_PREFIX", "SMARTTILE_PREFIX"] {
        if let Ok(p) = std::env::var(var) {
            out.push(PathBuf::from(p));
        }
    }
    out.push(PathBuf::from("/opt/smarttile"));
    out
}

fn load() -> Result<Api, String> {
    let mut candidates: Vec<PathBuf> = std::env::var("SMARTTILE_LIBPROJ").map(|p| vec![PathBuf::from(p)]).unwrap_or_default();
    for prefix in prefixes() {
        candidates.push(prefix.join("lib/libproj.so"));
    }
    candidates.push(PathBuf::from("libproj.so"));
    let mut errors = Vec::new();
    for candidate in candidates {
        match unsafe { Library::new(&candidate) } {
            Ok(lib) => unsafe {
                macro_rules! sym { ($name:literal) => { *lib.get($name).map_err(|e| format!("libproj {}: {e}", String::from_utf8_lossy($name)))? } }
                return Ok(Api {
                    context_create: sym!(b"proj_context_create"),
                    context_set_search_paths: sym!(b"proj_context_set_search_paths"),
                    crs_to_crs: sym!(b"proj_create_crs_to_crs"),
                    normalize: sym!(b"proj_normalize_for_visualization"),
                    trans: sym!(b"proj_trans"),
                    destroy: sym!(b"proj_destroy"),
                    errno: sym!(b"proj_errno"),
                    errno_string: sym!(b"proj_context_errno_string"),
                    create: sym!(b"proj_create"),
                    as_wkt: sym!(b"proj_as_wkt"),
                    get_type: sym!(b"proj_get_type"),
                    get_name: sym!(b"proj_get_name"),
                    source_crs: sym!(b"proj_get_source_crs"),
                    sub_crs: sym!(b"proj_crs_get_sub_crs"),
                    coordinate_system: sym!(b"proj_crs_get_coordinate_system"),
                    axis_count: sym!(b"proj_cs_get_axis_count"),
                    axis_info: sym!(b"proj_cs_get_axis_info"),
                    identify: sym!(b"proj_identify"),
                    list_count: sym!(b"proj_list_get_count"),
                    list_get: sym!(b"proj_list_get"),
                    list_destroy: sym!(b"proj_list_destroy"),
                    int_list_destroy: sym!(b"proj_int_list_destroy"),
                    id_auth_name: sym!(b"proj_get_id_auth_name"),
                    id_code: sym!(b"proj_get_id_code"),
                    is_equivalent: sym!(b"proj_is_equivalent_to"),
                    _lib: lib,
                });
            },
            Err(e) => errors.push(format!("{}: {e}", candidate.display())),
        }
    }
    Err(format!("libproj not found ({})", errors.join("; ")))
}

fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(load).as_ref().map_err(Clone::clone)
}

fn context(api: &Api) -> *mut c_void {
    unsafe {
        let ctx = (api.context_create)();
        if std::env::var_os("PROJ_DATA").is_none() && std::env::var_os("PROJ_LIB").is_none() {
            if let Some(dir) = prefixes().into_iter().map(|p| p.join("share/proj")).find(|p| p.join("proj.db").exists()) {
                let dir = CString::new(dir.to_string_lossy().as_bytes()).unwrap();
                let paths = [dir.as_ptr()];
                (api.context_set_search_paths)(ctx, 1, paths.as_ptr());
            }
        }
        ctx
    }
}

/// `pyproj.CRS(definition).to_wkt()`: single-line WKT2_2019.
pub fn wkt2(definition: &str) -> Result<String, String> {
    let api = api()?;
    unsafe {
        let ctx = context(api);
        let text = CString::new(definition).unwrap();
        let pj = (api.create)(ctx, text.as_ptr());
        if pj.is_null() {
            return Err(format!("PROJ cannot read CRS {definition}"));
        }
        let option = CString::new("MULTILINE=NO").unwrap();
        let options = [option.as_ptr(), std::ptr::null()];
        let wkt = (api.as_wkt)(ctx, pj, 2, options.as_ptr());
        let out = if wkt.is_null() { Err(format!("PROJ cannot write {definition} as WKT")) }
                  else { Ok(CStr::from_ptr(wkt).to_string_lossy().into_owned()) };
        (api.destroy)(pj);
        out
    }
}

/// A CRS-to-CRS transformation with x/y in easting/northing, longitude/latitude
/// order (pyproj `Transformer.from_crs(..., always_xy=True)`).
pub struct Transformer {
    ctx: *mut c_void,
    pj: *mut c_void,
}

impl Transformer {
    pub fn new(source: &str, target: &str) -> Result<Transformer, String> {
        let api = api()?;
        unsafe {
            let ctx = context(api);
            let (s, t) = (CString::new(source).unwrap(), CString::new(target).unwrap());
            let raw = (api.crs_to_crs)(ctx, s.as_ptr(), t.as_ptr(), std::ptr::null_mut());
            if raw.is_null() {
                let message = CStr::from_ptr((api.errno_string)(ctx, (api.errno)(std::ptr::null()))).to_string_lossy().into_owned();
                return Err(format!("PROJ cannot transform {source} to {target}: {message}"));
            }
            let pj = (api.normalize)(ctx, raw);
            (api.destroy)(raw);
            if pj.is_null() {
                return Err(format!("PROJ cannot normalize {source} -> {target}"));
            }
            Ok(Transformer { ctx, pj })
        }
    }

    pub fn transform(&self, x: f64, y: f64) -> (f64, f64) {
        let api = api().expect("loaded");
        let out = unsafe { (api.trans)(self.pj, 1, Coord([x, y, 0.0, f64::INFINITY])) };
        (out.0[0], out.0[1])
    }
}

impl Drop for Transformer {
    fn drop(&mut self) {
        if let Ok(api) = api() {
            unsafe { (api.destroy)(self.pj); }
        }
        let _ = self.ctx; // the context lives for the process
    }
}


fn text(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() { None } else { Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()) }
}

// PJ_TYPE values of proj.h.
const GEOCENTRIC: c_int = 10;
const GEOGRAPHIC: [c_int; 3] = [11, 12, 13];
const PROJECTED: c_int = 15;
const COMPOUND: c_int = 16;
const BOUND: c_int = 19;

pub struct Axis { pub name: String, pub unit_name: String, pub unit_factor: f64 }

/// A CRS with pyproj's semantics (`pyproj.CRS`).
pub struct Crs { ctx: *mut c_void, pj: *mut c_void }

unsafe impl Send for Crs {}
unsafe impl Sync for Crs {}

impl Drop for Crs {
    fn drop(&mut self) {
        if let Ok(api) = api() {
            unsafe { (api.destroy)(self.pj); }
        }
    }
}

impl Crs {
    /// `pyproj.CRS(definition)`: WKT, "EPSG:<code>" or any PROJ input.
    pub fn new(definition: &str) -> Result<Crs, String> {
        let api = api()?;
        let ctx = context(api);
        let c = CString::new(definition.trim_end_matches('\0')).map_err(|_| "CRS text contains NUL".to_string())?;
        let pj = unsafe { (api.create)(ctx, c.as_ptr()) };
        if pj.is_null() {
            return Err(format!("Invalid CRS: {}", definition.chars().take(80).collect::<String>()));
        }
        Ok(Crs { ctx, pj })
    }

    pub fn from_epsg(code: u32) -> Result<Crs, String> {
        Crs::new(&format!("EPSG:{code}"))
    }

    fn wrap(&self, pj: *mut c_void) -> Option<Crs> {
        if pj.is_null() { None } else { Some(Crs { ctx: self.ctx, pj }) }
    }

    fn kind(&self) -> c_int {
        unsafe { (api().unwrap().get_type)(self.pj) }
    }

    pub fn name(&self) -> String {
        text(unsafe { (api().unwrap().get_name)(self.pj) }).unwrap_or_default()
    }

    pub fn is_bound(&self) -> bool { self.kind() == BOUND }
    pub fn is_compound(&self) -> bool { self.kind() == COMPOUND }
    pub fn is_projected(&self) -> bool { self.kind() == PROJECTED }
    pub fn is_geographic(&self) -> bool { GEOGRAPHIC.contains(&self.kind()) }
    pub fn is_geocentric(&self) -> bool { self.kind() == GEOCENTRIC }

    pub fn source_crs(&self) -> Option<Crs> {
        self.wrap(unsafe { (api().unwrap().source_crs)(self.ctx, self.pj) })
    }

    pub fn sub_crs_list(&self) -> Vec<Crs> {
        (0..).map_while(|i| self.wrap(unsafe { (api().unwrap().sub_crs)(self.ctx, self.pj, i) })).collect()
    }

    /// The source of a bound CRS, else itself.
    pub fn unbound(self) -> Crs {
        if self.is_bound() { self.source_crs().unwrap_or(self) } else { self }
    }

    /// `CRS.axis_info`: the coordinate system's axes (compound: every part's; bound: its source's).
    pub fn axis_info(&self) -> Vec<Axis> {
        if self.is_compound() {
            return self.sub_crs_list().iter().flat_map(|c| c.axis_info()).collect();
        }
        if self.is_bound() {
            return self.source_crs().map(|c| c.axis_info()).unwrap_or_default();
        }
        let api = api().unwrap();
        let Some(cs) = self.wrap(unsafe { (api.coordinate_system)(self.ctx, self.pj) }) else { return Vec::new() };
        let count = unsafe { (api.axis_count)(self.ctx, cs.pj) };
        (0..count).filter_map(|i| unsafe {
            let (mut name, mut abbrev, mut direction, mut unit_name, mut unit_auth, mut unit_code) =
                (std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null());
            let mut factor = 0.0;
            if (api.axis_info)(self.ctx, cs.pj, i, &mut name, &mut abbrev, &mut direction, &mut factor, &mut unit_name, &mut unit_auth, &mut unit_code) == 0 {
                return None;
            }
            Some(Axis { name: text(name).unwrap_or_default(), unit_name: text(unit_name).unwrap_or_default(), unit_factor: factor })
        }).collect()
    }

    /// `CRS.to_authority(auth_name, min_confidence)`: the best identification, if confident enough.
    pub fn to_authority(&self, auth_name: Option<&str>, min_confidence: i32) -> Option<(String, String)> {
        let api = api().ok()?;
        let auth = auth_name.map(|a| CString::new(a).unwrap());
        let mut confidence: *mut c_int = std::ptr::null_mut();
        unsafe {
            let list = (api.identify)(self.ctx, self.pj, auth.as_ref().map_or(std::ptr::null(), |a| a.as_ptr()), std::ptr::null(), &mut confidence);
            if list.is_null() {
                return None;
            }
            let count = (api.list_count)(list);
            let mut out = None;
            if count > 0 && !confidence.is_null() && *confidence >= min_confidence {
                let first = (api.list_get)(self.ctx, list, 0);
                if !first.is_null() {
                    if let (Some(a), Some(c)) = (text((api.id_auth_name)(first, 0)), text((api.id_code)(first, 0))) {
                        out = Some((a, c));
                    }
                    (api.destroy)(first);
                }
            }
            if !confidence.is_null() {
                (api.int_list_destroy)(confidence);
            }
            (api.list_destroy)(list);
            out
        }
    }

    pub fn to_epsg(&self, min_confidence: i32) -> Option<u32> {
        self.to_authority(Some("EPSG"), min_confidence).and_then(|(_, code)| code.parse().ok())
    }

    /// `crs_authority_from_crs`: "AUTH:code" from to_authority(), else EPSG.
    pub fn authority_string(&self) -> Option<String> {
        if let Some((auth, code)) = self.to_authority(None, 70) {
            if !auth.is_empty() && !code.is_empty() {
                return Some(format!("{}:{code}", auth.to_uppercase()));
            }
        }
        self.to_epsg(70).map(|code| format!("EPSG:{code}"))
    }

    /// `CRS.to_wkt(version)`, single line. Versions: "WKT2_2019", "WKT1_GDAL".
    pub fn to_wkt(&self, version: &str) -> Result<String, String> {
        let kind = match version { "WKT2_2019" => 2, "WKT1_GDAL" => 4, other => return Err(format!("WKT version {other}")) };
        let option = CString::new("MULTILINE=NO").unwrap();
        let options = [option.as_ptr(), std::ptr::null()];
        text(unsafe { (api()?.as_wkt)(self.ctx, self.pj, kind, options.as_ptr()) }).ok_or_else(|| format!("CRS cannot be written as {version}"))
    }

    /// `CRS.equals(other, ignore_axis_order=True)`.
    pub fn equals_ignoring_axis_order(&self, other: &Crs) -> bool {
        unsafe { (api().unwrap().is_equivalent)(self.pj, other.pj, 2) != 0 }
    }
}

/// `copc_metadata.crs_equivalent`.
pub fn crs_equivalent(a: &Crs, b: &Crs) -> bool {
    if let (Some(x), Some(y)) = (a.authority_string(), b.authority_string()) {
        if x == y {
            return true;
        }
    }
    a.equals_ignoring_axis_order(b) || matches!((a.to_wkt("WKT2_2019"), b.to_wkt("WKT2_2019")), (Ok(x), Ok(y)) if x == y)
}
