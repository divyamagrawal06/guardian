"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Search, Zap, Loader2, CheckCircle2, XCircle, ChevronRight, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Griffin components
import { BlueprintCanvas } from "@/components/griffin/blueprint-canvas";
import { ChatPage } from "@/components/griffin/chat-page";
import { GodModeTerminal } from "@/components/griffin/god-mode-terminal";
import { Workstation } from "@/components/griffin/workstation";
import { CostDashboard } from "@/components/griffin/cost-dashboard";
import { MultiverseScene } from "@/components/griffin/multiverse";
import { Navbar as GriffinNavbar } from "@/components/griffin/navbar";
import { SettingsDialog } from "@/components/griffin/settings-dialog";
import { useOrchestratorStore } from "@/lib/griffin/orchestrator-store";

// Types
type AppMode = "atlas" | "turbo";
type TurboView = "canvas" | "chat" | "terminal" | "workstation" | "cost" | "multiverse";

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

const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const ATLAS_WS_URL = "ws://localhost:8000/ws";
const GRIFFIN_WS_URL = process.env.NEXT_PUBLIC_ML_SERVICE_URL ?? "ws://localhost:9100";

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

// ATLAS WebSocket hook
function useAtlasAgent() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ProgressMessage[]>([]);
  const [result, setResult] = useState<ProgressMessage | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    try {
      const ws = new WebSocket(ATLAS_WS_URL);
      ws.onopen = () => {};
      ws.onmessage = (event) => {
        const data: ProgressMessage = JSON.parse(event.data);
        if (data.type === "connected") {
          setConnected(true);
        } else if (data.type === "progress") {
          setProgress((prev) => [...prev, data]);
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
        setTimeout(connect, 2000);
      };
      ws.onerror = () => setConnected(false);
      wsRef.current = ws;
    } catch {
      console.log("ATLAS backend not available");
    }
  }, []);

  useEffect(() => {
    connect();
    return () => { wsRef.current?.close(); };
  }, [connect]);

  const sendCommand = useCallback((command: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
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

// Mode Pill Switcher
function ModeSwitcher({ mode, onChange }: { mode: AppMode; onChange: (m: AppMode) => void }) {
  return (
    <div className="fixed top-3 left-1/2 -translate-x-1/2 z-[100] flex items-center gap-1 bg-black/60 backdrop-blur-xl border border-white/10 rounded-full px-1.5 py-1 shadow-2xl">
      <button
        onClick={() => onChange("atlas")}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 ${
          mode === "atlas"
            ? "bg-white/15 text-white shadow-sm"
            : "text-white/40 hover:text-white/70"
        }`}
      >
        <Zap className="w-3 h-3" />
        ATLAS
      </button>
      <button
        onClick={() => onChange("turbo")}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 ${
          mode === "turbo"
            ? "bg-gradient-to-r from-[#fa0f83] to-[#911150] text-white shadow-lg shadow-[#fa0f83]/30"
            : "text-white/40 hover:text-white/70"
        }`}
      >
        <Sparkles className="w-3 h-3" />
        Super Turbo
      </button>
    </div>
  );
}

// ATLAS Agent View
function AtlasView() {
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<IndexedItem[]>([]);
  const [filtered, setFiltered] = useState<IndexedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [mode, setMode] = useState<"search" | "agent">("search");
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const agent = useAtlasAgent();

  useEffect(() => {
    async function loadItems() {
      try {
        const result = await invoke<IndexedItem[]>("get_indexed_items");
        setItems(result || []);
        setFiltered(result || []);
      } catch {
        // no-op in web mode
      } finally {
        setLoading(false);
      }
    }
    loadItems();
  }, []);

  useEffect(() => {
    listenEvent("focus-search", () => {
      inputRef.current?.focus();
      setQuery("");
      setMode("search");
    });
  }, []);

  useEffect(() => {
    if (query.trim() === "") {
      setFiltered(items);
      setMode("search");
    } else {
      const q = query.toLowerCase();
      setFiltered(items.filter((item) => item.name.toLowerCase().includes(q)));
    }
    setSelectedIndex(0);
  }, [query, items]);

  const openItem = useCallback(async (item: IndexedItem) => {
    try { await invoke("open_item", { path: item.path }); } catch {}
  }, []);

  const submitCommand = useCallback(() => {
    if (query.trim() && agent.connected) agent.sendCommand(query.trim());
  }, [query, agent]);

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
        if (mode === "agent") submitCommand();
        else if (filtered[selectedIndex]) openItem(filtered[selectedIndex]);
      } else if (e.key === "Escape") {
        if (agent.running) agent.stopAgent();
        else if (isTauri) {
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

  useEffect(() => {
    const listEl = listRef.current;
    if (listEl) {
      const selected = listEl.children[selectedIndex] as HTMLElement;
      if (selected) selected.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  const apps = filtered.filter((i) => i.kind === "app");
  const files = filtered.filter((i) => i.kind !== "app");
  const displayItems = [...apps, ...files];
  let globalIndex = -1;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.15, ease: "easeOut" }}
      className="flex flex-col w-full h-screen bg-[#0c0508]/90 backdrop-blur-md overflow-hidden border border-white/5 shadow-2xl relative"
      onKeyDown={handleKeyDown}
    >
      <div className="relative z-10 flex flex-col h-full pt-12">
        <div className="border-b border-[#911150]/30">
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
              {agent.connected && <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />}
            </button>
            <span className="ml-auto text-[10px] text-[#e693bc]/25">Tab to switch</span>
          </div>
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
              <button onClick={() => setQuery("")} className="text-[#e693bc]/40 hover:text-[#fa0f83] text-sm transition-colors">?</button>
            )}
            {agent.running && (
              <button onClick={agent.stopAgent} className="text-red-400 hover:text-red-300 text-xs transition-colors">Stop</button>
            )}
          </div>
        </div>

        <div ref={listRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
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
              <AnimatePresence>
                {agent.progress.map((p, i) => (
                  <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex items-start gap-2 text-sm px-2 py-1.5">
                    {p.status === "started" || p.status === "planning" || p.status === "executing" ? (
                      <Loader2 className="w-3.5 h-3.5 mt-0.5 text-[#fa0f83] animate-spin shrink-0" />
                    ) : p.status === "completed" ? (
                      <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 text-green-400 shrink-0" />
                    ) : p.status === "failed" ? (
                      <XCircle className="w-3.5 h-3.5 mt-0.5 text-red-400 shrink-0" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5 mt-0.5 text-[#e693bc]/40 shrink-0" />
                    )}
                    <div>
                      <span className="text-[#e693bc]/60 text-xs uppercase tracking-wide">{p.step}</span>
                      {p.detail && <p className="text-[#f0e4ea]/70 text-xs mt-0.5">{p.detail}</p>}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              {agent.result && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex items-center gap-2 px-3 py-3 rounded-lg text-sm mt-2 ${
                    agent.result.success ? "bg-green-400/10 text-green-300" : "bg-red-400/10 text-red-300"
                  }`}
                >
                  {agent.result.success ? <CheckCircle2 className="w-4 h-4 shrink-0" /> : <XCircle className="w-4 h-4 shrink-0" />}
                  {agent.result.detail || (agent.result.success ? "Task completed" : "Task failed")}
                </motion.div>
              )}
            </div>
          )}

          {mode === "search" && (
            <>
              {loading && (
                <div className="text-[#e693bc]/30 text-sm text-center py-8">
                  <Loader2 className="w-5 h-5 animate-spin mx-auto mb-2" />
                  Loading...
                </div>
              )}
              {!loading && displayItems.length === 0 && query && (
                <div className="text-[#e693bc]/30 text-sm text-center py-8">No results for &ldquo;{query}&rdquo;</div>
              )}
              {displayItems.map((item) => {
                globalIndex++;
                const idx = globalIndex;
                const isSelected = idx === selectedIndex;
                return (
                  <motion.div
                    key={item.path}
                    initial={false}
                    animate={{ backgroundColor: isSelected ? "rgba(145,17,80,0.25)" : "transparent" }}
                    transition={{ duration: 0.1 }}
                    className="flex items-center gap-3 px-3 py-2.5 rounded-xl cursor-pointer group"
                    onClick={() => openItem(item)}
                  >
                    <span className="text-lg leading-none">{item.icon}</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-[#f0e4ea] text-sm truncate">{item.name}</p>
                      <p className="text-[#e693bc]/30 text-xs truncate">{item.path}</p>
                    </div>
                    <span className="text-[#e693bc]/20 text-xs group-hover:text-[#e693bc]/50 transition-colors capitalize">{item.kind}</span>
                  </motion.div>
                );
              })}
            </>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// Super Turbo (Griffin) View
function TurboView() {
  const [activeView, setActiveView] = useState<TurboView>("canvas");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const connect = useOrchestratorStore((s) => s.connect);
  const connected = useOrchestratorStore((s) => s.connected);

  useEffect(() => {
    if (!connected) {
      connect(GRIFFIN_WS_URL);
    }
  }, [connect, connected]);

  const viewComponents: Record<TurboView, React.ComponentType<any>> = {
    canvas: BlueprintCanvas,
    chat: ChatPage,
    terminal: GodModeTerminal,
    workstation: Workstation,
    cost: CostDashboard,
    multiverse: () => (
      <MultiverseScene
        isOpen={true}
        onClose={() => {}}
        onSelectUniverse={(_universe: any, _index: number) => {
          setTimeout(() => setActiveView("canvas"), 100);
        }}
      />
    ),
  };

  const ActiveComponent = viewComponents[activeView];

  return (
    <div className="turbo-mode h-screen w-full bg-background overflow-hidden">
      <GriffinNavbar
        activeView={activeView}
        onViewChange={(v) => setActiveView(v as TurboView)}
        onSettingsClick={() => setSettingsOpen(true)}
      />
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />

      <div className="relative h-full w-full overflow-hidden pb-[68px] pt-12">
        <div
          className="absolute inset-0 w-full h-full"
          style={{
            opacity: activeView === "multiverse" ? 1 : 0,
            pointerEvents: activeView === "multiverse" ? "auto" : "none",
            zIndex: activeView === "multiverse" ? 10 : 0,
          }}
        >
          <MultiverseScene
            isOpen={true}
            onClose={() => {}}
            onSelectUniverse={(_universe: any, _index: number) => {
              setTimeout(() => setActiveView("canvas"), 100);
            }}
          />
        </div>

        {activeView !== "multiverse" && (
          <AnimatePresence mode="wait">
            <motion.div
              key={activeView}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
              className="relative h-full w-full"
              style={{ zIndex: 20 }}
            >
              <ActiveComponent />
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}

// Root
export default function Home() {
  const [appMode, setAppMode] = useState<AppMode>("atlas");

  return (
    <>
      <ModeSwitcher mode={appMode} onChange={setAppMode} />
      <AnimatePresence mode="wait">
        {appMode === "atlas" ? (
          <motion.div
            key="atlas"
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 12 }}
            transition={{ duration: 0.25 }}
            className="h-screen w-full"
          >
            <AtlasView />
          </motion.div>
        ) : (
          <motion.div
            key="turbo"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
            className="h-screen w-full"
          >
            <TurboView />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}