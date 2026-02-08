"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Search } from "lucide-react";
import { motion } from "framer-motion";

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

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) => Math.min(prev + 1, filtered.length - 1));
      } else if (e.key === "ArrowUp") {
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
        selected.scrollIntoView({ block: "nearest" });
      }
    }
  }, [selectedIndex]);

  // Group: apps first, then files
  const apps = filtered.filter((i) => i.kind === "app");
  const files = filtered.filter((i) => i.kind !== "app");
  const displayItems = [...apps, ...files];

  let globalIndex = -1;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.15, ease: "easeOut" }}
      className="flex flex-col w-full h-screen bg-[#0c0508]/90 backdrop-blur-md rounded-2xl overflow-hidden border border-white/5 shadow-2xl relative"
      onKeyDown={handleKeyDown}
    >
      {/* Content wrapper */}
      <div className="relative z-10 flex flex-col h-full">
        {/* Search Bar */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-[#911150]/30">
          <Search className="w-5 h-5 text-[#e693bc]/50 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search apps and files..."
            autoFocus
            className="flex-1 bg-transparent text-[#f0e4ea] text-base placeholder:text-[#e693bc]/30 outline-none"
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              className="text-[#e693bc]/40 hover:text-[#fa0f83] text-sm transition-colors"
            >
              ✕
            </button>
          )}
        </div>

        {/* Results */}
        <div ref={listRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-1 scrollbar-thin scrollbar-thumb-[#911150]/20">
          {loading ? (
            <div className="flex items-center justify-center h-32">
              <div className="text-[#e693bc]/50 text-sm">Indexing files...</div>
            </div>
          ) : displayItems.length === 0 ? (
            <div className="flex items-center justify-center h-32">
              <div className="text-[#e693bc]/50 text-sm">No results found</div>
            </div>
          ) : (
            <>
              {apps.length > 0 && (
                <div className="px-2 pt-2 pb-1">
                  <span className="text-[11px] font-medium text-[#e693bc]/40 uppercase tracking-wider">
                    Applications
                  </span>
                </div>
              )}
              {apps.map((item) => {
                globalIndex++;
                const idx = globalIndex;
                return (
                  <button
                    key={`app-${item.path}`}
                    onClick={() => openItem(item)}
                    className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-left transition-all duration-150 ${selectedIndex === idx
                      ? "bg-[#911150]/30 text-[#f0e4ea]"
                      : "text-[#e693bc]/70 hover:bg-[#911150]/15 hover:text-[#f0e4ea]"
                      }`}
                  >
                    <span className="text-lg shrink-0">{item.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">
                        {item.name}
                      </div>
                    </div>
                    <span className="text-[10px] text-[#fa0f83]/40 uppercase tracking-wider shrink-0">
                      {item.kind}
                    </span>
                  </button>
                );
              })}

              {files.length > 0 && (
                <div className="px-2 pt-3 pb-1">
                  <span className="text-[11px] font-medium text-[#e693bc]/40 uppercase tracking-wider">
                    Files
                  </span>
                </div>
              )}
              {files.map((item) => {
                globalIndex++;
                const idx = globalIndex;
                return (
                  <button
                    key={`file-${item.path}`}
                    onClick={() => openItem(item)}
                    className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-left transition-all duration-150 ${selectedIndex === idx
                      ? "bg-[#911150]/30 text-[#f0e4ea]"
                      : "text-[#e693bc]/70 hover:bg-[#911150]/15 hover:text-[#f0e4ea]"
                      }`}
                  >
                    <span className="text-lg shrink-0">{item.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">
                        {item.name}
                      </div>
                      <div className="text-[11px] text-[#e693bc]/25 truncate">
                        {item.path}
                      </div>
                    </div>
                    <span className="text-[10px] text-[#fa0f83]/40 uppercase tracking-wider shrink-0">
                      {item.kind}
                    </span>
                  </button>
                );
              })}
            </>
          )}
        </div>
      </div> {/* Close content wrapper */}
    </motion.div>
  );
}
