"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Zap, Loader2, CheckCircle2, XCircle, ChevronRight, Sparkles, Mic, Settings } from "lucide-react";
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

interface ProgressMessage {
  type: string;
  step?: string;
  status?: string;
  detail?: string;
  success?: boolean;
  message?: string;
}

const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
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

// -- ATLAS WebSocket hook ------------------------------------------------------
function useAtlasAgent(url: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const urlRef = useRef(url);
  urlRef.current = url;

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ProgressMessage[]>([]);
  const [result, setResult] = useState<ProgressMessage | null>(null);

  const connect = useCallback(() => {
    wsRef.current?.close();
    try {
      const ws = new WebSocket(urlRef.current);
      ws.onopen = () => setConnected(true);
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
        const savedWs = ws;
        setTimeout(() => {
          if (wsRef.current === savedWs) connect();
        }, 2000);
      };
      ws.onerror = () => setConnected(false);
      wsRef.current = ws;
    } catch {
      setTimeout(connect, 3000);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Reconnect when URL changes
  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [url]); // eslint-disable-line react-hooks/exhaustive-deps

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

// -- Mode Pill Switcher --------------------------------------------------------
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

// -- ATLAS View (Griffin dark-pink theme) --------------------------------------
function AtlasView() {
  const [query, setQuery] = useState("");
  const [listening, setListening] = useState(false);
  const [showConn, setShowConn] = useState(false);
  const [wsHost, setWsHost] = useState("localhost");
  const [wsPort, setWsPort] = useState("8000");

  const wsUrl = `ws://${wsHost}:${wsPort}/ws`;
  const agent = useAtlasAgent(wsUrl);

  const inputRef = useRef<HTMLInputElement>(null);
  const feedRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  // Auto-scroll feed on new progress
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [agent.progress, agent.result]);

  // Voice input via Web Speech API
  const toggleVoice = useCallback(() => {
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }
    const SpeechRec =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SpeechRec) return;
    const rec = new SpeechRec();
    rec.continuous = false;
    rec.interimResults = true;
    rec.onresult = (e: any) => {
      const transcript = Array.from(e.results as any[])
        .map((r: any) => r[0].transcript)
        .join("");
      setQuery(transcript);
      if (e.results[e.results.length - 1].isFinal) {
        setListening(false);
        if (transcript.trim() && agent.connected) {
          agent.sendCommand(transcript.trim());
        }
      }
    };
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    rec.start();
    recognitionRef.current = rec;
    setListening(true);
  }, [listening, agent]);

  const submitCommand = useCallback(() => {
    if (query.trim() && agent.connected) {
      agent.sendCommand(query.trim());
    }
  }, [query, agent]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        submitCommand();
      } else if (e.key === "Escape") {
        if (agent.running) agent.stopAgent();
        else if (showConn) setShowConn(false);
      }
    },
    [submitCommand, agent, showConn]
  );

  const statusColor = agent.connected ? "#4ade80" : "#f97316";
  const statusText = agent.connected
    ? `Connected - ws://${wsHost}:${wsPort}`
    : `Connecting to ws://${wsHost}:${wsPort}...`;

  return (
    <div
      className="relative w-full h-screen bg-[#0c0508] overflow-hidden flex flex-col items-center"
      style={{
        backgroundImage: `
          radial-gradient(ellipse 90% 55% at 50% -5%, rgba(250,15,131,0.09) 0%, transparent 65%),
          radial-gradient(ellipse 45% 35% at 5% 95%, rgba(145,17,80,0.07) 0%, transparent 60%)
        `,
      }}
      onKeyDown={handleKeyDown}
    >
      {/* Dot-grid texture */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `radial-gradient(circle, rgba(230,147,188,0.11) 1px, transparent 1px)`,
          backgroundSize: "28px 28px",
          opacity: 0.65,
        }}
      />
      {/* Subtle scanlines */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.025]"
        style={{
          backgroundImage: `repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(250,15,131,0.6) 2px, rgba(250,15,131,0.6) 3px)`,
        }}
      />

      {/* Content column */}
      <div className="relative z-10 w-full max-w-xl px-5 pt-20 flex flex-col items-center flex-1 min-h-0">

        {/* -- Header: wordmark -- */}
        <motion.div
          initial={{ opacity: 0, y: -18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="flex flex-col items-center mb-8 select-none"
        >
          <h1
            className="text-[2.1rem] font-black text-transparent bg-clip-text leading-tight"
            style={{
              backgroundImage: "linear-gradient(140deg, #f0e4ea 10%, #fa0f83 50%, #e693bc 90%)",
              fontFamily: "'Monocraft', 'Courier New', monospace",
              letterSpacing: "0.28em",
            }}
          >
            ATLAS
          </h1>
          <p className="text-[#e693bc]/38 text-[9.5px] tracking-[0.38em] uppercase mt-1.5 font-mono">
            Autonomous Desktop Agent
          </p>
        </motion.div>

        {/* -- Command card -- */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.12 }}
          className="w-full rounded-2xl border border-[#911150]/40 bg-[#0f0306]/92 backdrop-blur-md overflow-hidden"
          style={{
            boxShadow: agent.running
              ? "0 0 0 1px rgba(250,15,131,0.35), 0 0 55px rgba(250,15,131,0.12), 0 24px 64px rgba(0,0,0,0.65)"
              : "0 0 0 1px rgba(145,17,80,0.2), 0 24px 64px rgba(0,0,0,0.55)",
          }}
        >
          {/* Header bar */}
          <div className="flex items-center px-3 pt-2.5 pb-2 border-b border-[#911150]/20">
            <div className="flex items-center gap-1.5 text-[#fa0f83] text-xs font-medium">
              <Zap className="w-3 h-3" />
              agent
            </div>
            {agent.connected && (
              <div className="ml-auto flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-green-400 shadow-[0_0_6px_#4ade80]" />
                <span className="text-[9px] text-green-400/60 font-mono">live</span>
              </div>
            )}
          </div>

          {/* Input row */}
          <div className="flex items-center gap-2 px-4 py-3.5">
            <Zap className="w-4 h-4 text-[#fa0f83]/50 shrink-0" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoFocus
              disabled={agent.running}
              placeholder={
                agent.running
                  ? "Agent is working..."
                  : "Tell ATLAS what to do..."
              }
              className="flex-1 bg-transparent text-[#f0e4ea] text-sm placeholder:text-[#e693bc]/20 outline-none disabled:opacity-40"
            />
            {query && !agent.running && (
              <button
                onClick={() => setQuery("")}
                className="text-[#e693bc]/25 hover:text-[#e693bc]/55 transition-colors text-xs leading-none"
              >
                ?
              </button>
            )}
            {/* Mic button */}
            <button
              onClick={toggleVoice}
              className={`p-1.5 rounded-lg transition-all ${
                listening
                  ? "text-[#fa0f83] bg-[#fa0f83]/12 shadow-[0_0_12px_rgba(250,15,131,0.4)] animate-pulse"
                  : "text-[#e693bc]/26 hover:text-[#e693bc]/58 hover:bg-white/[0.04]"
              }`}
              title={listening ? "Stop listening" : "Voice input"}
            >
              <Mic className="w-3.5 h-3.5" />
            </button>
            {/* Send / Stop */}
            {agent.running ? (
              <button
                onClick={agent.stopAgent}
                className="px-2.5 py-1.5 rounded-lg text-xs font-mono bg-red-500/10 text-red-400/70 hover:bg-red-500/18 border border-red-500/22 transition-all"
              >
                stop
              </button>
            ) : (
              <button
                onClick={submitCommand}
                disabled={!query.trim() || !agent.connected}
                className="p-1.5 rounded-lg bg-gradient-to-br from-[#fa0f83] to-[#911150] text-white disabled:opacity-22 disabled:cursor-not-allowed hover:shadow-[0_0_16px_rgba(250,15,131,0.48)] active:scale-95 transition-all"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Status bar */}
          <div className="flex items-center gap-2 px-4 pb-3 pt-0.5">
            <div
              className="w-1.5 h-1.5 rounded-full shrink-0 transition-all"
              style={{
                backgroundColor: statusColor,
                boxShadow: agent.connected ? `0 0 6px ${statusColor}` : "none",
              }}
            />
            <span className="text-[10px] text-[#e693bc]/30 flex-1 truncate font-mono">
              {statusText}
            </span>
            <button
              onClick={() => setShowConn((v) => !v)}
              className="p-1 rounded text-[#e693bc]/20 hover:text-[#e693bc]/55 transition-colors"
              title="Configure connection"
            >
              <Settings className="w-3 h-3" />
            </button>
          </div>
        </motion.div>

        {/* -- Feed / search results -- */}
        <div ref={feedRef} className="w-full mt-3 flex-1 overflow-y-auto space-y-1.5 pb-6 min-h-0">

          {/* Agent mode */}
              {!agent.connected && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center gap-2.5 px-4 py-3 rounded-xl bg-[#fa0f83]/5 border border-[#fa0f83]/12 text-[#e693bc]/42 text-xs font-mono"
                >
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-[#fa0f83]/48 shrink-0" />
                  Waiting for backend at ws://{wsHost}:{wsPort}
                </motion.div>
              )}

              {agent.connected && !agent.running && agent.progress.length === 0 && !agent.result && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="flex flex-col items-center py-10 gap-3"
                >
                  <img
                    src="/peeking-owl.png"
                    alt=""
                    className="w-10 h-10 object-contain opacity-[0.18]"
                    style={{ imageRendering: "pixelated" }}
                  />
                  <p className="text-[#e693bc]/20 text-xs text-center font-mono leading-loose">
                    Type a command and press{" "}
                    <kbd className="px-1.5 py-0.5 rounded bg-[#911150]/22 text-[#fa0f83]/55 text-[10px] not-italic font-mono">
                      Enter
                    </kbd>
                    <br />
                    <span className="text-[#e693bc]/12 text-[10px]">
                      e.g. &ldquo;Open Notepad and type hello world&rdquo;
                    </span>
                  </p>
                </motion.div>
              )}

              <AnimatePresence>
                {agent.progress.map((p, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 5, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.18 }}
                    className="flex items-start gap-3 px-4 py-3 rounded-xl bg-[#0f0306]/80 border border-[#911150]/16 backdrop-blur-sm"
                  >
                    <div className="mt-0.5 shrink-0">
                      {p.status === "completed" ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-green-400/75" />
                      ) : p.status === "failed" ? (
                        <XCircle className="w-3.5 h-3.5 text-red-400/75" />
                      ) : (
                        <Loader2 className="w-3.5 h-3.5 text-[#fa0f83]/65 animate-spin" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                        <span className="text-[9px] font-mono tracking-widest text-[#fa0f83]/52 uppercase">
                          {p.step}
                        </span>
                        {p.status && (
                          <span
                            className={`text-[9px] font-mono uppercase tracking-wide px-1.5 py-0.5 rounded-full ${
                              p.status === "completed"
                                ? "bg-green-500/10 text-green-400/65"
                                : p.status === "failed"
                                ? "bg-red-500/10 text-red-400/65"
                                : "bg-[#fa0f83]/8 text-[#fa0f83]/58"
                            }`}
                          >
                            {p.status}
                          </span>
                        )}
                      </div>
                      {p.detail && (
                        <p className="text-[#f0e4ea]/42 text-xs font-mono leading-relaxed break-words">
                          {p.detail}
                        </p>
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {agent.result && (
                <motion.div
                  initial={{ opacity: 0, y: 8, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  className={`flex items-start gap-3 px-4 py-4 rounded-xl border ${
                    agent.result.success
                      ? "bg-green-500/[0.06] border-green-500/20"
                      : "bg-red-500/[0.06] border-red-500/20"
                  }`}
                >
                  {agent.result.success ? (
                    <CheckCircle2 className="w-4 h-4 text-green-400/75 shrink-0 mt-0.5" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-400/75 shrink-0 mt-0.5" />
                  )}
                  <div>
                    <p
                      className={`text-xs font-mono font-medium ${
                        agent.result.success ? "text-green-300/75" : "text-red-300/75"
                      }`}
                    >
                      {agent.result.success ? "Task completed successfully" : "Task failed"}
                    </p>
                    {agent.result.detail && (
                      <p
                        className={`text-xs font-mono mt-1 leading-relaxed ${
                          agent.result.success ? "text-green-300/45" : "text-red-300/45"
                        }`}
                      >
                        {agent.result.detail}
                      </p>
                    )}
                  </div>
                </motion.div>
              )}

        </div>
      </div>

      {/* -- Connection panel (slide-in) -- */}
      <AnimatePresence>
        {showConn && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 bg-black/45 backdrop-blur-sm"
              onClick={() => setShowConn(false)}
            />
            <motion.aside
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", damping: 26, stiffness: 220 }}
              className="fixed right-0 top-0 bottom-0 w-72 z-50 flex flex-col shadow-2xl"
              style={{
                background: "#0a0208",
                borderLeft: "1px solid rgba(250,15,131,0.15)",
              }}
            >
              {/* Panel header */}
              <div className="flex items-start justify-between px-5 py-4 border-b border-[#911150]/20">
                <div>
                  <h2 className="text-[#f0e4ea]/88 text-sm font-semibold font-mono">
                    Configure Backend
                  </h2>
                  <p className="text-[#e693bc]/32 text-[10px] mt-0.5 font-mono">
                    ATLAS WebSocket connection
                  </p>
                </div>
                <button
                  onClick={() => setShowConn(false)}
                  className="text-[#e693bc]/28 hover:text-[#e693bc]/65 transition-colors mt-0.5"
                >
                  <XCircle className="w-4 h-4" />
                </button>
              </div>

              {/* Panel body */}
              <div className="flex-1 overflow-y-auto px-5 py-5 space-y-5">
                <p className="text-[#e693bc]/36 text-[11px] leading-relaxed font-mono">
                  Same Wi-Fi: enter your PC&apos;s local IP.
                  <br />
                  USB tunnel:{" "}
                  <code className="text-[#fa0f83]/50 text-[10px]">
                    adb reverse tcp:8000 tcp:8000
                  </code>
                  {" "}then use{" "}
                  <code className="text-[#fa0f83]/50 text-[10px]">localhost</code>.
                </p>

                <div className="space-y-1.5">
                  <label className="text-[#e693bc]/38 text-[9px] uppercase tracking-widest font-mono block">
                    Host
                  </label>
                  <input
                    type="text"
                    value={wsHost}
                    onChange={(e) => setWsHost(e.target.value)}
                    placeholder="localhost"
                    className="w-full rounded-lg px-3 py-2.5 text-[#f0e4ea] text-xs font-mono outline-none transition-colors placeholder:text-[#e693bc]/18"
                    style={{
                      background: "#0f0306",
                      border: "1px solid rgba(145,17,80,0.28)",
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.borderColor = "rgba(250,15,131,0.48)";
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = "rgba(145,17,80,0.28)";
                    }}
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-[#e693bc]/38 text-[9px] uppercase tracking-widest font-mono block">
                    Port
                  </label>
                  <input
                    type="number"
                    value={wsPort}
                    onChange={(e) => setWsPort(e.target.value)}
                    placeholder="8000"
                    className="w-full rounded-lg px-3 py-2.5 text-[#f0e4ea] text-xs font-mono outline-none transition-colors placeholder:text-[#e693bc]/18"
                    style={{
                      background: "#0f0306",
                      border: "1px solid rgba(145,17,80,0.28)",
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.borderColor = "rgba(250,15,131,0.48)";
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = "rgba(145,17,80,0.28)";
                    }}
                  />
                </div>

                {/* Preview */}
                <div
                  className="rounded-lg px-3 py-2 text-[10px] font-mono flex items-center gap-0.5 flex-wrap"
                  style={{ background: "rgba(250,15,131,0.04)", border: "1px solid rgba(145,17,80,0.2)" }}
                >
                  <span className="text-[#fa0f83]/45">ws://</span>
                  <span className="text-[#f0e4ea]/50">{wsHost || "localhost"}</span>
                  <span className="text-[#e693bc]/28">:</span>
                  <span className="text-[#f0e4ea]/50">{wsPort || "8000"}</span>
                  <span className="text-[#fa0f83]/45">/ws</span>
                </div>
              </div>

              {/* Panel footer */}
              <div className="px-5 pb-5 pt-3 border-t border-[#911150]/18 space-y-2">
                <button
                  onClick={() => setShowConn(false)}
                  className="w-full py-2.5 rounded-xl text-white text-xs font-medium font-mono active:scale-[0.98] transition-all"
                  style={{
                    background: "linear-gradient(135deg, #fa0f83, #911150)",
                    boxShadow: "0 0 0 1px rgba(250,15,131,0.3)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.boxShadow = "0 0 20px rgba(250,15,131,0.38), 0 0 0 1px rgba(250,15,131,0.4)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow = "0 0 0 1px rgba(250,15,131,0.3)";
                  }}
                >
                  Apply & Reconnect
                </button>
                <p className="text-center text-[#e693bc]/16 text-[9px] font-mono">
                  connection updates automatically
                </p>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

// -- Super Turbo (Griffin) View ------------------------------------------------
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

// -- Root ----------------------------------------------------------------------
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