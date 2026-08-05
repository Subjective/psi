//! The five default tools: canonical arguments, declared effects, bounded
//! output, and workspace-root enforcement (Milestone 2).

use std::fs;
use std::path::{Path, PathBuf};

use psi_core::tool::{ToolEffect, ToolInvocation, ToolOutput, ToolRegistry};
use psi_core::tools::default_profile;
use serde_json::json;
use tempfile::TempDir;

/// A fixture repository: a couple of source files, a nested directory, and a
/// `.git` directory that no walk should ever descend into.
fn fixture() -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::create_dir_all(root.join(".git/objects")).unwrap();
    fs::write(
        root.join("README.md"),
        "# fixture\n\nA sample repository.\n",
    )
    .unwrap();
    fs::write(
        root.join("src/lib.rs"),
        "pub fn answer() -> u32 {\n    41\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("src/main.rs"),
        "fn main() {\n    println!(\"hi\");\n}\n",
    )
    .unwrap();
    fs::write(root.join(".git/objects/pack"), "binary junk\n").unwrap();
    dir
}

async fn run(
    tools: &ToolRegistry,
    root: &Path,
    name: &str,
    arguments: serde_json::Value,
) -> ToolOutput {
    tools
        .get(name)
        .unwrap_or_else(|| panic!("{name} is not in the profile"))
        .execute(ToolInvocation {
            call_id: format!("call-{name}"),
            arguments,
            cwd: root.to_path_buf(),
        })
        .await
}

#[test]
fn the_profile_declares_the_effects_the_revision_rules_need() {
    let tools = default_profile(PathBuf::from("/fixture"));
    let effects: Vec<(String, ToolEffect)> = [
        "read_file",
        "list_directory",
        "search",
        "apply_patch",
        "exec",
    ]
    .iter()
    .map(|name| {
        let tool = tools.get(name).expect("advertised");
        (tool.spec().name, tool.effect())
    })
    .collect();
    assert_eq!(
        effects,
        vec![
            ("read_file".to_string(), ToolEffect::ReadOnly),
            ("list_directory".to_string(), ToolEffect::ReadOnly),
            ("search".to_string(), ToolEffect::ReadOnly),
            ("apply_patch".to_string(), ToolEffect::Mutating),
            ("exec".to_string(), ToolEffect::Unknown),
        ]
    );
    // Advertisement order is the profile order; the predictor sees the same.
    let advertised: Vec<String> = tools.specs().into_iter().map(|spec| spec.name).collect();
    assert_eq!(
        advertised,
        [
            "read_file",
            "list_directory",
            "search",
            "apply_patch",
            "exec"
        ]
    );
}

#[tokio::test]
async fn read_file_returns_exact_text_and_line_ranges() {
    let dir = fixture();
    let tools = default_profile(dir.path().to_path_buf());

    let whole = run(
        &tools,
        dir.path(),
        "read_file",
        json!({ "path": "src/lib.rs" }),
    )
    .await;
    assert_eq!(whole.content, "pub fn answer() -> u32 {\n    41\n}\n");
    assert!(whole.error.is_none());
    assert!(!whole.truncated);

    let slice = run(
        &tools,
        dir.path(),
        "read_file",
        json!({ "path": "src/lib.rs", "start_line": 2, "line_count": 1 }),
    )
    .await;
    assert_eq!(slice.content, "    41");

    let missing = run(
        &tools,
        dir.path(),
        "read_file",
        json!({ "path": "nope.rs" }),
    )
    .await;
    assert!(missing.error.is_some());
}

#[tokio::test]
async fn read_file_bounds_its_output() {
    let dir = fixture();
    fs::write(dir.path().join("big.txt"), "x".repeat(100 * 1024)).unwrap();
    let tools = default_profile(dir.path().to_path_buf());

    let output = run(
        &tools,
        dir.path(),
        "read_file",
        json!({ "path": "big.txt" }),
    )
    .await;
    assert!(output.truncated);
    assert!(output.content.contains("[truncated:"));
    assert!(output.content.len() < 100 * 1024);
}

#[tokio::test]
async fn list_directory_descends_on_request_and_never_into_git() {
    let dir = fixture();
    let tools = default_profile(dir.path().to_path_buf());

    let shallow = run(&tools, dir.path(), "list_directory", json!({})).await;
    assert_eq!(shallow.content, "README.md\nsrc/");

    let deep = run(&tools, dir.path(), "list_directory", json!({ "depth": 3 })).await;
    assert_eq!(deep.content, "README.md\nsrc/\nsrc/lib.rs\nsrc/main.rs");
    assert!(!deep.content.contains(".git"));
}

#[tokio::test]
async fn search_reports_matches_by_path_and_line() {
    let dir = fixture();
    let tools = default_profile(dir.path().to_path_buf());

    let hits = run(
        &tools,
        dir.path(),
        "search",
        json!({ "pattern": "fn \\w+\\(" }),
    )
    .await;
    assert_eq!(
        hits.content,
        "src/lib.rs:1: pub fn answer() -> u32 {\nsrc/main.rs:1: fn main() {"
    );

    let none = run(
        &tools,
        dir.path(),
        "search",
        json!({ "pattern": "nowhere" }),
    )
    .await;
    assert_eq!(none.content, "no matches for nowhere");
    assert!(none.error.is_none());

    let bad = run(&tools, dir.path(), "search", json!({ "pattern": "(" })).await;
    assert!(bad.error.is_some());
}

#[tokio::test]
async fn apply_patch_replaces_unique_text_and_creates_files() {
    let dir = fixture();
    let tools = default_profile(dir.path().to_path_buf());

    let edit = run(
        &tools,
        dir.path(),
        "apply_patch",
        json!({ "path": "src/lib.rs", "old_text": "    41\n", "new_text": "    42\n" }),
    )
    .await;
    assert!(edit.error.is_none(), "{edit:?}");
    assert_eq!(
        fs::read_to_string(dir.path().join("src/lib.rs")).unwrap(),
        "pub fn answer() -> u32 {\n    42\n}\n"
    );

    let created = run(
        &tools,
        dir.path(),
        "apply_patch",
        json!({ "path": "src/new.rs", "old_text": "", "new_text": "pub fn added() {}\n" }),
    )
    .await;
    assert!(created.error.is_none());
    assert_eq!(
        fs::read_to_string(dir.path().join("src/new.rs")).unwrap(),
        "pub fn added() {}\n"
    );

    // Creating over an existing file, and editing text that is not unique or
    // not present, all fail with something the model can act on.
    let clobber = run(
        &tools,
        dir.path(),
        "apply_patch",
        json!({ "path": "src/new.rs", "old_text": "", "new_text": "x" }),
    )
    .await;
    assert!(clobber.error.unwrap().contains("already exists"));

    let absent = run(
        &tools,
        dir.path(),
        "apply_patch",
        json!({ "path": "src/lib.rs", "old_text": "nowhere", "new_text": "x" }),
    )
    .await;
    assert!(absent.error.unwrap().contains("not in the file"));

    fs::write(dir.path().join("dup.txt"), "same\nsame\n").unwrap();
    let ambiguous = run(
        &tools,
        dir.path(),
        "apply_patch",
        json!({ "path": "dup.txt", "old_text": "same\n", "new_text": "x\n" }),
    )
    .await;
    assert!(ambiguous.error.unwrap().contains("appears 2 times"));
}

#[tokio::test]
async fn exec_reports_output_and_exit_status_and_kills_on_timeout() {
    let dir = fixture();
    let tools = default_profile(dir.path().to_path_buf());

    let ok = run(&tools, dir.path(), "exec", json!({ "command": "echo hi" })).await;
    assert_eq!(ok.content, "hi\n[exit status: 0]");
    assert!(ok.error.is_none());

    // A failing command is an answer, not a broken tool: the model has to be
    // able to read failing test output.
    let failing = run(
        &tools,
        dir.path(),
        "exec",
        json!({ "command": "echo boom >&2; exit 3" }),
    )
    .await;
    assert!(failing.error.is_none());
    assert!(failing.content.contains("[stderr]\nboom"));
    assert!(failing.content.contains("[exit status: 3]"));

    let slow = run(
        &tools,
        dir.path(),
        "exec",
        json!({ "command": "sleep 30", "timeout_ms": 100 }),
    )
    .await;
    assert!(slow.error.unwrap().contains("killed after 100ms"));

    // exec runs from the workspace root but is not confined to it by design.
    let outside = run(&tools, dir.path(), "exec", json!({ "command": "ls .." })).await;
    assert!(outside.error.is_none());
}

#[tokio::test]
async fn structured_tools_refuse_paths_outside_the_workspace_root() {
    let dir = fixture();
    let outside = dir.path().parent().unwrap().join("outside.txt");
    fs::write(&outside, "secret\n").unwrap();
    let tools = default_profile(dir.path().to_path_buf());

    for arguments in [
        json!({ "path": "../outside.txt" }),
        json!({ "path": "src/../../outside.txt" }),
        json!({ "path": outside.to_string_lossy() }),
    ] {
        let refused = run(&tools, dir.path(), "read_file", arguments.clone()).await;
        assert!(
            refused
                .error
                .as_deref()
                .is_some_and(|error| error.contains("escapes the workspace root")),
            "read_file {arguments} was not refused: {refused:?}"
        );
    }

    let listed = run(
        &tools,
        dir.path(),
        "list_directory",
        json!({ "path": ".." }),
    )
    .await;
    assert!(listed.error.unwrap().contains("escapes the workspace root"));

    let searched = run(
        &tools,
        dir.path(),
        "search",
        json!({ "pattern": "secret", "path": ".." }),
    )
    .await;
    assert!(
        searched
            .error
            .unwrap()
            .contains("escapes the workspace root")
    );

    let patched = run(
        &tools,
        dir.path(),
        "apply_patch",
        json!({ "path": "../outside.txt", "old_text": "secret", "new_text": "leaked" }),
    )
    .await;
    assert!(
        patched
            .error
            .unwrap()
            .contains("escapes the workspace root")
    );
    assert_eq!(fs::read_to_string(&outside).unwrap(), "secret\n");

    fs::remove_file(outside).unwrap();
}

#[tokio::test]
async fn a_symlink_out_of_the_workspace_is_refused() {
    let dir = fixture();
    let outside = dir.path().parent().unwrap().join("linked.txt");
    fs::write(&outside, "secret\n").unwrap();
    std::os::unix::fs::symlink(&outside, dir.path().join("link.txt")).unwrap();
    let tools = default_profile(dir.path().to_path_buf());

    let refused = run(
        &tools,
        dir.path(),
        "read_file",
        json!({ "path": "link.txt" }),
    )
    .await;
    assert!(
        refused
            .error
            .unwrap()
            .contains("escapes the workspace root")
    );

    fs::remove_file(outside).unwrap();
}

#[tokio::test]
async fn arguments_outside_the_advertised_schema_are_rejected() {
    let dir = fixture();
    let tools = default_profile(dir.path().to_path_buf());

    let extra = run(
        &tools,
        dir.path(),
        "read_file",
        json!({ "path": "README.md", "encoding": "utf-8" }),
    )
    .await;
    assert!(extra.error.unwrap().contains("invalid arguments"));

    let missing = run(&tools, dir.path(), "read_file", json!({})).await;
    assert!(missing.error.unwrap().contains("invalid arguments"));
}
