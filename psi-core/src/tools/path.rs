//! Workspace-root enforcement for the structured file tools. `exec` is exempt
//! by design: it inherits Psi's process permissions (docs/design.md, "Trusted
//! environment and hooks").

use std::path::{Component, Path, PathBuf};

/// Resolves the root once at construction so every later containment check
/// compares canonical paths. A root that cannot be canonicalized — it may not
/// exist yet — is kept as given.
pub fn canonical_root(root: PathBuf) -> PathBuf {
    root.canonicalize().unwrap_or(root)
}

/// Resolves a path argument against the workspace root and rejects anything
/// that leaves it. Two checks, because neither alone is enough: the lexically
/// normalized path must stay under the root, and the nearest existing ancestor
/// must still be under the root once symlinks are resolved.
pub fn resolve_in_root(root: &Path, raw: &str) -> Result<PathBuf, String> {
    let joined = if Path::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        root.join(raw)
    };
    let normalized = normalize(&joined);
    if !normalized.starts_with(root) {
        return Err(escape(raw));
    }

    let mut existing = normalized.as_path();
    while !existing.exists() {
        match existing.parent() {
            Some(parent) => existing = parent,
            None => return Err(escape(raw)),
        }
    }
    match existing.canonicalize() {
        Ok(resolved) if resolved.starts_with(root) => Ok(normalized),
        _ => Err(escape(raw)),
    }
}

fn escape(raw: &str) -> String {
    format!("path escapes the workspace root: {raw}")
}

/// Resolves `.` and `..` without touching the filesystem. It cannot see
/// through symlinks, which is why `resolve_in_root` also canonicalizes.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}
