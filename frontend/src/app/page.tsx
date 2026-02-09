"use client";

import { useEffect, useRef, useState, useCallback } from "react";

// Types
interface IndexedItem {
  name: string;
  path: string;
  kind: string;
  icon: string;
}

// Detect if running in Tauri
const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  console.log("invoke called:", cmd, "isTauri:", isTauri);
  if (isTauri) {
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      const result = await tauriInvoke<T>(cmd, args);
      console.log("Tauri invoke result:", result);
      return result;
    } catch (e) {
      console.error("Tauri invoke error:", e);
      throw e;
    }
  }
  console.log("Not in Tauri, returning empty array");
  return [] as unknown as T;
}

async function listenEvent(event: string, handler: () => void) {
  if (isTauri) {
    const { listen } = await import("@tauri-apps/api/event");
    return listen(event, handler);
  }
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<IndexedItem[]>([]);
  const [filtered, setFiltered] = useState<IndexedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Load indexed items on mount
  useEffect(() => {
    async function loadItems() {
      try {
        console.log("Loading items from Tauri...");
        const result = await invoke<IndexedItem[]>("get_indexed_items");
        console.log("Received items:", result?.length, result);
        setItems(result || []);
        setFiltered(result || []);
      } catch (e) {
        console.error("Failed to load items:", e);
      } finally {
        setLoading(false);
      }
    }
    loadItems();
  }, []);

  // Listen for focus-search event from Tauri (on hotkey toggle)
  useEffect(() => {
    listenEvent("focus-search", () => {
      inputRef.current?.focus();
      setQuery("");
    });
  }, []);

  // Filter items on query change
  useEffect(() => {
    if (query.trim() === "") {
      setFiltered(items);
    } else {
      const q = query.toLowerCase();
      setFiltered(
        items.filter((item) => item.name.toLowerCase().includes(q))
      );
    }
    setSelectedIndex(0);
  }, [query, items]);

  // Open the selected item
  const openItem = useCallback(
    async (item: IndexedItem) => {
      try {
        await invoke("open_item", { path: item.path });
      } catch (e) {
        console.error("Failed to open item:", e);
      }
    },
    []
  );

  // Keyboard navigation (horizontal layout)
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) => Math.min(prev + 1, filtered.length - 1));
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === "Enter" && filtered[selectedIndex]) {
        e.preventDefault();
        openItem(filtered[selectedIndex]);
      } else if (e.key === "Escape") {
        if (isTauri) {
          import("@tauri-apps/api/window").then(({ getCurrentWindow }) => {
            getCurrentWindow().hide();
          });
        }
      }
    },
    [filtered, selectedIndex, openItem]
  );

  // Auto-scroll selected item into view
  useEffect(() => {
    const listEl = listRef.current;
    if (listEl) {
      const selected = listEl.children[selectedIndex] as HTMLElement;
      if (selected) {
        selected.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
    }
  }, [selectedIndex]);

  // Get display items (show all for horizontal scroll)
  const displayItems = filtered;

  return (
    <div
      className="w-full h-screen flex flex-col relative overflow-hidden"
      onKeyDown={handleKeyDown}
    >

      {/* Window Frame */}
      <div className="pixel-window flex-1 flex flex-col">
        {/* Title Bar */}
        <div className="pixel-titlebar">
          <span className="pixel-title">ATLAS</span>
          <div className="pixel-controls">
            <button className="pixel-btn-settings" title="Settings">
              <span>⚙</span>
            </button>
            <button className="pixel-btn-close" title="Close">
              <span>✕</span>
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="pixel-content flex-1 flex flex-col">
          {/* Search Bar */}
          <div className="pixel-searchbar">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search anything..."
              autoFocus
              className="pixel-search-input"
            />
            <span className="pixel-search-icon">🔍</span>
          </div>

          {/* Items Grid */}
          <div
            ref={listRef}
            className="pixel-grid flex-1 overflow-auto"
          >
            {loading ? (
              <div className="col-span-5 flex items-center justify-center h-24">
                <span className="pixel-text pixel-loading">Loading...</span>
              </div>
            ) : displayItems.length === 0 ? (
              <div className="col-span-5 flex items-center justify-center h-24">
                <span className="pixel-text">No items found</span>
              </div>
            ) : (
              displayItems.map((item, index) => (
                <button
                  key={`${item.kind}-${item.path}`}
                  onClick={() => openItem(item)}
                  className={`pixel-item ${selectedIndex === index ? 'selected' : ''}`}
                >
                  <div className="pixel-item-icon">
                    <img
                      src="/folder.svg"
                      alt={item.name}
                      className="folder-icon"
                    />
                  </div>
                  <span className="pixel-item-name">{item.name}</span>
                </button>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
