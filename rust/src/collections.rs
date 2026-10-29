//! `--collection-dir` (`collection_folders.py`): every product of a task
//! hard-linked into one flat folder per output collection, named
//! `<collection>/<file stem>.<last extension>` for the Galaxy wrapper.

use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use crate::lasio::Res;

pub fn file_stem(path: &Path) -> String {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    if name.to_lowercase().ends_with(".copc.laz") {
        return name[..name.len() - ".copc.laz".len()].to_string();
    }
    Path::new(&name).file_stem().unwrap().to_string_lossy().into_owned()
}

fn existing_ancestor(path: &Path) -> PathBuf {
    let mut path = std::path::absolute(path).unwrap_or(path.to_path_buf());
    while !path.exists() {
        path = path.parent().map(Path::to_path_buf).unwrap_or(PathBuf::from("/"));
    }
    path
}

/// `check_collection_dir`: new or empty, and on the outputs' filesystem (hard links).
pub fn check(collection_dir: &Path, output_root: &Path) -> Res<()> {
    let nonempty = collection_dir.is_dir() && fs::read_dir(collection_dir).map(|mut d| d.next().is_some()).unwrap_or(true);
    if collection_dir.is_symlink() || (collection_dir.exists() && (!collection_dir.is_dir() || nonempty)) {
        return Err(format!("--collection-dir must be a new or empty directory: {}", collection_dir.display()));
    }
    let device = |p: &Path| fs::metadata(existing_ancestor(p)).map(|m| m.dev()).map_err(|e| e.to_string());
    if device(collection_dir)? != device(output_root)? {
        return Err(format!("--collection-dir must be on the same filesystem as the outputs (hard links): {}", collection_dir.display()));
    }
    Ok(())
}

/// Top-level regular files of `folder` with one of `extensions` (lower-case, with dot).
pub fn files(folder: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(folder) else { return Vec::new() };
    let mut out: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with('.'))
        .filter(|p| p.extension().is_some_and(|e| extensions.contains(&format!(".{}", e.to_string_lossy().to_lowercase()).as_str())))
        .collect();
    out.sort();
    out
}

/// `tile_collections`: tile output i (`subsampled_res<i>`) is collection `tiles_<i>`.
pub fn tile_collections(output_dir: &Path) -> Vec<(String, Vec<PathBuf>)> {
    let mut out = Vec::new();
    let mut index = 1;
    while output_dir.join(format!("subsampled_res{index}")).is_dir() {
        out.push((format!("tiles_{index}"), files(&output_dir.join(format!("subsampled_res{index}")), &[".laz", ".las", ".ply"])));
        index += 1;
    }
    out
}

/// `publish_collections`: hard-link every product; nothing stays published if a link fails.
pub fn publish(collection_dir: &Path, collections: &[(String, Vec<PathBuf>)], single: &[(&str, PathBuf)]) -> Res<()> {
    let mut products = Vec::new();
    for (collection, paths) in collections {
        let mut seen: Vec<(String, &PathBuf)> = Vec::new();
        for path in paths {
            let designation = file_stem(path);
            if let Some((_, other)) = seen.iter().find(|(d, _)| *d == designation.to_lowercase()) {
                return Err(format!("{collection}: {} and {} would both be named {designation}",
                                   path.file_name().unwrap().to_string_lossy(), other.file_name().unwrap().to_string_lossy()));
            }
            seen.push((designation.to_lowercase(), path));
            let suffix = path.extension().map(|e| format!(".{}", e.to_string_lossy().to_lowercase())).unwrap_or_default();
            products.push((path.clone(), collection_dir.join(collection).join(format!("{designation}{suffix}"))));
        }
    }
    products.extend(single.iter().filter(|(_, p)| p.is_file()).map(|(name, p)| (p.clone(), collection_dir.join(name))));
    fs::create_dir_all(collection_dir).map_err(|e| e.to_string())?;
    for (collection, _) in collections {
        fs::create_dir_all(collection_dir.join(collection)).map_err(|e| e.to_string())?;
    }
    let mut installed = Vec::new();
    for (source, destination) in &products {
        if let Err(e) = fs::hard_link(source, destination) {
            for path in installed {
                let _ = fs::remove_file(path);
            }
            return Err(format!("{}: {e}", destination.display()));
        }
        installed.push(destination);
    }
    let mut names: Vec<&str> = collections.iter().map(|(c, _)| c.as_str()).collect();
    names.sort();
    println!("Collection folders: {} ({})", collection_dir.display(), if names.is_empty() { "none".to_string() } else { names.join(", ") });
    Ok(())
}
