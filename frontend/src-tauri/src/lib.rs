use serde::Serialize;
use std::path::PathBuf;
use tauri::{Emitter, Manager};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize)]
pub struct IndexedItem {
    pub name: String,
    pub path: String,
    pub kind: String, // "app", "file", "folder", "image", "video", "document"
    pub icon: String, // emoji or icon hint
}

fn detect_kind(path: &std::path::Path, is_app: bool) -> (String, String) {
    if is_app {
        return ("app".into(), "🖥️".into());
    }
    if path.is_dir() {
        return ("folder".into(), "📁".into());
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "svg" | "webp" | "ico" => {
            ("image".into(), "🖼️".into())
        }
        "mp4" | "mkv" | "avi" | "mov" | "webm" | "flv" => ("video".into(), "🎬".into()),
        "mp3" | "wav" | "flac" | "ogg" | "aac" => ("audio".into(), "🎵".into()),
        "pdf" | "doc" | "docx" | "txt" | "rtf" | "odt" | "xls" | "xlsx" | "ppt" | "pptx" => {
            ("document".into(), "📄".into())
        }
        "zip" | "rar" | "7z" | "tar" | "gz" => ("archive".into(), "📦".into()),
        "exe" | "msi" => ("app".into(), "🖥️".into()),
        "lnk" => ("shortcut".into(), "🔗".into()),
        _ => ("file".into(), "📄".into()),
    }
}

fn scan_user_folders() -> Vec<IndexedItem> {
    let mut items = Vec::new();

    if let Some(user_dir) = dirs::home_dir() {
        let folders = ["Desktop", "Downloads", "Pictures", "Videos"];

        for folder_name in &folders {
            let folder_path = user_dir.join(folder_name);
            if !folder_path.exists() {
                continue;
            }

            // Walk max 2 levels deep to keep it fast
            for entry in WalkDir::new(&folder_path)
                .max_depth(2)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();

                // Skip the root folder itself
                if path == folder_path {
                    continue;
                }

                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();

                if name.starts_with('.') {
                    continue;
                }

                let (kind, icon) = detect_kind(path, false);

                items.push(IndexedItem {
                    name,
                    path: path.to_string_lossy().to_string(),
                    kind,
                    icon,
                });
            }
        }
    }

    items
}

fn scan_apps() -> Vec<IndexedItem> {
    let mut items = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Scan common app locations on Windows
    let app_dirs: Vec<PathBuf> = vec![
        // Start Menu - All Users
        PathBuf::from(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs"),
        // Start Menu - Current User
        dirs::home_dir()
            .map(|h| {
                h.join(r"AppData\Roaming\Microsoft\Windows\Start Menu\Programs")
            })
            .unwrap_or_default(),
        // Desktop shortcuts
        dirs::home_dir()
            .map(|h| h.join("Desktop"))
            .unwrap_or_default(),
    ];

    for app_dir in app_dirs {
        if !app_dir.exists() {
            continue;
        }

        for entry in WalkDir::new(&app_dir)
            .max_depth(3)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            if ext == "lnk" || ext == "exe" {
                let name = path
                    .file_stem()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();

                if name.is_empty() || seen.contains(&name.to_lowercase()) {
                    continue;
                }
                seen.insert(name.to_lowercase());

                items.push(IndexedItem {
                    name,
                    path: path.to_string_lossy().to_string(),
                    kind: "app".into(),
                    icon: "🖥️".into(),
                });
            }
        }
    }

    items
}

#[tauri::command]
fn get_indexed_items() -> Vec<IndexedItem> {
    let mut all = scan_apps();
    all.extend(scan_user_folders());
    all
}

#[tauri::command]
fn search_items(query: String) -> Vec<IndexedItem> {
    let q = query.to_lowercase();
    let all = get_indexed_items();

    if q.is_empty() {
        return all;
    }

    all.into_iter()
        .filter(|item| item.name.to_lowercase().contains(&q))
        .collect()
}

#[tauri::command]
fn open_item(path: String) -> Result<(), String> {
    open::that(&path).map_err(|e| format!("Failed to open {}: {}", path, e))
}

fn toggle_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        if window.is_visible().unwrap_or(false) {
            let _ = window.hide();
        } else {
            let _ = window.show();
            let _ = window.set_focus();
            let _ = window.center();
            // Tell the frontend to focus the search input
            let _ = window.emit("focus-search", ());
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Register Win+- global shortcut
            let shortcut = Shortcut::new(Some(Modifiers::SUPER), Code::Minus);
            let handle = app.handle().clone();

            app.handle().plugin(
                tauri_plugin_global_shortcut::Builder::new()
                    .with_handler(move |_app, _shortcut, event| {
                        if event.state == tauri_plugin_global_shortcut::ShortcutState::Pressed {
                            toggle_window(&handle);
                        }
                    })
                    .build(),
            )?;

            // Try to register the shortcut, but don't fail if it's already registered
            if let Err(e) = app.global_shortcut().register(shortcut) {
                eprintln!("Warning: Failed to register global shortcut: {}", e);
            }

            // Also hide window when it loses focus
            let window = app.get_webview_window("main").unwrap();
            let w = window.clone();
            window.on_window_event(move |event| {
                if let tauri::WindowEvent::Focused(false) = event {
                    let _ = w.hide();
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_indexed_items,
            search_items,
            open_item
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
