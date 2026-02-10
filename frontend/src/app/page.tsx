"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Search, Zap, Loader2, CheckCircle2, XCircle, ChevronRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Types
interface IndexedItem {
  name: string;
  path: string;
  kind: string;
  icon: string;
}

interface ProgressMessage {
  type: string;
  step?: string;
  status?: string;
  detail?: string;
  success?: boolean;
  message?: string;
}

// Detect if running in Tauri
const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

const WS_URL = "ws://localhost:8000/ws";

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (isTauri) {
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      return await tauriInvoke<T>(cmd, args);
    } catch (e) {
      console.error("Tauri invoke error:", e);
      throw e;
    }
  }
  return [] as unknown as T;
}

async function listenEvent(event: string, handler: () => void) {
  if (isTauri) {
    const { listen } = await import("@tauri-apps/api/event");
    return listen(event, handler);
  }
}

// ── WebSocket Hook ──────────────────────────────────────────────────────────

function useAtlasAgent() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ProgressMessage[]>([]);
  const [result, setResult] = useState<ProgressMessage | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    
    try {
      const ws = new WebSocket(WS_URL);
      
      ws.onopen = () => {
        console.log("WebSocket connected to ATLAS backend");
      };
      
      ws.onmessage = (event) => {
        const data: ProgressMessage = JSON.parse(event.data);
        
        if (data.type === "connected") {
          setConnected(true);
        } else if (data.type === "progress") {
          setProgress(prev => [...prev, data]);
        } else if (data.type === "result") {
          setResult(data);
          setRunning(false);
        } else if (data.type === "error") {
          setResult({ type: "error", success: false, detail: data.message });
          setRunning(false);
        }
      };
      
      ws.onclose = () => {
        setConnected(false);
        // Auto-reconnect after 2s
        setTimeout(connect, 2000);
      };
      
      ws.onerror = () => {
        setConnected(false);
      };
      
      wsRef.current = ws;
    } catch {
      console.log("Backend not available, running in search-only mode");
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  const sendCommand = useCallback((command: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error("WebSocket not connected");
      return;
    }
    setProgress([]);
    setResult(null);
    setRunning(true);
    wsRef.current.send(JSON.stringify({ type: "command", command }));
  }, []);

  const stopAgent = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
    }
    setRunning(false);
  }, []);

  return { connected, running, progress, result, sendCommand, stopAgent };
}

// ── Main Component ──────────────────────────────────────────────────────────

export default function Home() {
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<IndexedItem[]>([]);
  const [filtered, setFiltered] = useState<IndexedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [mode, setMode] = useState<"search" | "agent">("search");
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const agent = useAtlasAgent();

  // Load indexed items on mount
  useEffect(() => {
    async function loadItems() {
      try {
        const result = await invoke<IndexedItem[]>("get_indexed_items");
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

  // Listen for focus-search event from Tauri
  useEffect(() => {
    listenEvent("focus-search", () => {
      inputRef.current?.focus();
      setQuery("");
      setMode("search");
    });
  }, []);

  // Filter items + auto-detect mode
  useEffect(() => {
    if (query.trim() === "") {
      setFiltered(items);
      setMode("search");
    } else {
      const q = query.toLowerCase();
      const matches = items.filter((item) => item.name.toLowerCase().includes(q));
      setFiltered(matches);
      // Stay in whatever mode the user chose
    }
    setSelectedIndex(0);
  }, [query, items]);

  // Open the selected item
  const openItem = useCallback(async (item: IndexedItem) => {
    try {
      await invoke("open_item", { path: item.path });
    } catch (e) {
      console.error("Failed to open item:", e);
    }
  }, []);

  // Submit agent command
  const submitCommand = useCallback(() => {
    if (query.trim() && agent.connected) {
      agent.sendCommand(query.trim());
    }
  }, [query, agent]);

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) => Math.min(prev + 1, filtered.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (mode === "agent") {
          submitCommand();
        } else if (filtered[selectedIndex]) {
          openItem(filtered[selectedIndex]);
        }
      } else if (e.key === "Escape") {
        if (agent.running) {
          agent.stopAgent();
        } else if (isTauri) {
          import("@tauri-apps/api/window").then(({ getCurrentWindow }) => {
            getCurrentWindow().hide();
          });
        }
      } else if (e.key === "Tab") {
        e.preventDefault();
        setMode(mode === "search" ? "agent" : "search");
      }
    },
    [filtered, selectedIndex, openItem, mode, submitCommand, agent]
  );

  // Auto-scroll
  useEffect(() => {
    const listEl = listRef.current;
    if (listEl) {
      const selected = listEl.children[selectedIndex] as HTMLElement;
      if (selected) selected.scrollIntoView({ block: "nearest" });
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
      <div className="relative z-10 flex flex-col h-full">
        {/* Mode Tabs + Search Bar */}
        <div className="border-b border-[#911150]/30">
          {/* Mode Toggle */}
          <div className="flex items-center gap-1 px-4 pt-3">
            <button
              onClick={() => setMode("search")}
              className={`px-3 py-1 text-xs rounded-t-lg transition-colors ${
                mode === "search"
                  ? "bg-[#911150]/30 text-[#f0e4ea]"
                  : "text-[#e693bc]/40 hover:text-[#e693bc]/60"
              }`}
            >
              <Search className="w-3 h-3 inline mr-1" /> Search
            </button>
            <button
              onClick={() => setMode("agent")}
              className={`px-3 py-1 text-xs rounded-t-lg transition-colors flex items-center gap-1 ${
                mode === "agent"
                  ? "bg-[#911150]/30 text-[#f0e4ea]"
                  : "text-[#e693bc]/40 hover:text-[#e693bc]/60"
              }`}
            >
              <Zap className="w-3 h-3" /> Agent
              {agent.connected && (
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />
              )}
            </button>
            <span className="ml-auto text-[10px] text-[#e693bc]/25">Tab to switch</span>
          </div>

          {/* Input */}
          <div className="flex items-center gap-3 px-5 py-3">
            {mode === "search" ? (
              <Search className="w-5 h-5 text-[#e693bc]/50 shrink-0" />
            ) : (
              <Zap className="w-5 h-5 text-[#fa0f83]/60 shrink-0" />
            )}
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={mode === "search" ? "Search apps and files..." : "Tell ATLAS what to do..."}
              autoFocus
              className="flex-1 bg-transparent text-[#f0e4ea] text-base placeholder:text-[#e693bc]/30 outline-none"
            />
            {query && !agent.running && (
              <button
                onClick={() => setQuery("")}
                className="text-[#e693bc]/40 hover:text-[#fa0f83] text-sm transition-colors"
              >
                ✕
              </button>
            )}
            {agent.running && (
              <button
                onClick={agent.stopAgent}
                className="text-red-400 hover:text-red-300 text-xs transition-colors"
              >
                Stop
              </button>
            )}
          </div>
        </div>

        {/* Content Area */}
        <div ref={listRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-1 scrollbar-thin scrollbar-thumb-[#911150]/20">
          
          {/* ── Agent Mode ── */}
          {mode === "agent" && (
            <div className="px-2 py-3 space-y-2">
              {!agent.connected && (
                <div className="flex items-center gap-2 text-yellow-400/70 text-sm px-2 py-2 bg-yellow-400/5 rounded-lg">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Connecting to ATLAS backend...
                </div>
              )}
              
              {agent.connected && !agent.running && agent.progress.length === 0 && !agent.result && (
                <div className="text-[#e693bc]/40 text-sm text-center py-8">
                  Type a command and press Enter.<br />
                  <span className="text-[#e693bc]/25 text-xs">e.g. &ldquo;Open Notepad and type hello world&rdquo;</span>
                </div>
              )}

              {/* Progress Steps */}
              <AnimatePresence>
                {agent.progress.map((p, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-start gap-2 text-sm px-2 py-1.5"
                  >
                    {p.status === "started" || p.status === "planning" || p.status === "executing" ? (
                      <Loader2 className="w-3.5 h-3.5 mt-0.5 text-[#fa0f83] animate-spin shrink-0" />
                    ) : p.status === "completed" ? (
                      <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 text-green-400 shrink-0" />
                    ) : p.status === "failed" || p.status === "retrying" ? (
                      <XCircle className="w-3.5 h-3.5 mt-0.5 text-red-400 shrink-0" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5 mt-0.5 text-[#e693bc]/40 shrink-0" />
                    )}
                    <div>
                      <span className="text-[#e693bc]/60 uppercase text-[10px] tracking-wider">{p.step}</span>
                      <p className="text-[#f0e4ea]/80">{p.detail}</p>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {/* Final Result */}
              {agent.result && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex items-center gap-2 text-sm px-3 py-3 rounded-lg mt-2 ${
                    agent.result.success
                      ? "bg-green-400/10 text-green-300"
                      : "bg-red-400/10 text-red-300"
                  }`}
                >
                  {agent.result.success ? (
                    <CheckCircle2 className="w-4 h-4 shrink-0" />
                  ) : (
                    <XCircle className="w-4 h-4 shrink-0" />
                  )}
                  {agent.result.detail || (agent.result.success ? "Done!" : "Failed")}
                </motion.div>
              )}
            </div>
          )}

          {/* ── Search Mode ── */}
          {mode === "search" && (
            <>
              {loading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="text-[#e693bc]/50 text-sm">Indexing files...</div>
                </div>
              ) : displayItems.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-32 gap-2">
                  <div className="text-[#e693bc]/50 text-sm">No results found</div>
                  {agent.connected && query.trim() && (
                    <button
                      onClick={() => setMode("agent")}
                      className="text-[#fa0f83]/60 hover:text-[#fa0f83] text-xs transition-colors"
                    >
                      Try as agent command →
                    </button>
                  )}
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
                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-left transition-all duration-150 ${
                          selectedIndex === idx
                            ? "bg-[#911150]/30 text-[#f0e4ea]"
                            : "text-[#e693bc]/70 hover:bg-[#911150]/15 hover:text-[#f0e4ea]"
                        }`}
                      >
                        <span className="text-lg shrink-0">{item.icon}</span>
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium truncate">{item.name}</div>
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
                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-left transition-all duration-150 ${
                          selectedIndex === idx
                            ? "bg-[#911150]/30 text-[#f0e4ea]"
                            : "text-[#e693bc]/70 hover:bg-[#911150]/15 hover:text-[#f0e4ea]"
                        }`}
                      >
                        <span className="text-lg shrink-0">{item.icon}</span>
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium truncate">{item.name}</div>
                          <div className="text-[11px] text-[#e693bc]/25 truncate">{item.path}</div>
                        </div>
                        <span className="text-[10px] text-[#fa0f83]/40 uppercase tracking-wider shrink-0">
                          {item.kind}
                        </span>
                      </button>
                    );
                  })}
                </>
              )}
            </>
          )}
        </div>
      </div>
    </motion.div>
  );
}
