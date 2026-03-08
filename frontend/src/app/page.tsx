"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import {
  Zap, Loader2, CheckCircle2, XCircle, ChevronRight,
  Mic, Settings, LayoutGrid, Terminal, DollarSign, Layers,
  Code2, PanelRightClose, PanelRightOpen,
} from "lucide-react";
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform, type MotionValue } from "framer-motion";

// Griffin components
import { BlueprintCanvas } from "@/components/griffin/blueprint-canvas";
import { GodModeTerminal } from "@/components/griffin/god-mode-terminal";
import { Workstation } from "@/components/griffin/workstation";
import { CostDashboard } from "@/components/griffin/cost-dashboard";
import { MultiverseScene } from "@/components/griffin/multiverse";
import { SettingsDialog } from "@/components/griffin/settings-dialog";
import { useOrchestratorStore } from "@/lib/griffin/orchestrator-store";

// ── Constants ────────────────────────────────────────────────────────────────

const GRIFFIN_WS_URL = process.env.NEXT_PUBLIC_ML_SERVICE_URL ?? "ws://localhost:9100";

// ── Types ────────────────────────────────────────────────────────────────────

type RightTab = "blueprint" | "workstation" | "terminal" | "cost";

interface ProgressMessage {
  type: string;
  step?: string;
  status?: string;
  detail?: string;
  success?: boolean;
  message?: string;
}

// ── ATLAS WebSocket hook ─────────────────────────────────────────────────────

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

  useEffect(() => {
    connect();
    return () => { wsRef.current?.close(); };
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

// ── Dock Icon (macOS magnification) ──────────────────────────────────────────

function DockIcon({
  icon: Icon,
  label,
  isActive,
  onClick,
  mouseX,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  isActive: boolean;
  onClick: () => void;
  mouseX: MotionValue<number>;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const distance = useTransform(mouseX, (val: number) => {
    const bounds = ref.current?.getBoundingClientRect() ?? { x: 0, width: 0 };
    return val - bounds.x - bounds.width / 2;
  });
  const widthSync = useTransform(distance, [-120, 0, 120], [36, 52, 36]);
  const width = useSpring(widthSync, { mass: 0.1, stiffness: 200, damping: 14 });

  return (
    <motion.button
      ref={ref}
      style={{ width, height: width }}
      onClick={onClick}
      className={`group relative flex items-center justify-center rounded-xl transition-colors ${
        isActive
          ? "bg-[#fa0f83]/20 ring-1 ring-[#fa0f83]/40"
          : "bg-white/8 hover:bg-white/14"
      }`}
      whileTap={{ scale: 0.85 }}
      aria-label={label}
    >
      <Icon className={`w-[38%] h-[38%] ${isActive ? "text-[#fa0f83]" : "text-[#e693bc]/60"}`} />
      {isActive && (
        <motion.div
          layoutId="dock-dot"
          className="absolute -bottom-1.5 w-1 h-1 rounded-full bg-[#fa0f83]"
          transition={{ type: "spring", stiffness: 300, damping: 25 }}
        />
      )}
      <span className="pointer-events-none absolute -top-8 rounded-md bg-[#0a0208]/95 border border-[#911150]/30 px-2 py-1 text-[10px] text-[#e693bc] opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap backdrop-blur-md font-mono">
        {label}
      </span>
    </motion.button>
  );
}

// ── Main Unified View ────────────────────────────────────────────────────────

export default function Home() {
  // -- State --
  const [query, setQuery] = useState("");
  const [listening, setListening] = useState(false);
  const [showConn, setShowConn] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [wsHost, setWsHost] = useState("localhost");
  const [wsPort, setWsPort] = useState("8000");
  const [rightTab, setRightTab] = useState<RightTab>("blueprint");
  const [rightOpen, setRightOpen] = useState(true);
  const [multiverse, setMultiverse] = useState(false);

  // -- Connections --
  const wsUrl = `ws://${wsHost}:${wsPort}/ws`;
  const atlas = useAtlasAgent(wsUrl);

  const griffinConnect = useOrchestratorStore((s) => s.connect);
  const griffinConnected = useOrchestratorStore((s) => s.connected);
  const griffinPipelineRunning = useOrchestratorStore((s) => s.pipelineRunning);
  const griffinSend = useOrchestratorStore((s) => s.sendChatMessage);
  const griffinChat = useOrchestratorStore((s) => s.chatMessages);
  const griffinAgentMsgs = useOrchestratorStore((s) => s.agentMessages);
  const griffinWrappers = useOrchestratorStore((s) => s.wrappers);
  const griffinArtifacts = useOrchestratorStore((s) => s.artifacts);

  useEffect(() => {
    if (!griffinConnected) griffinConnect(GRIFFIN_WS_URL);
  }, [griffinConnect, griffinConnected]);

  // -- Refs --
  const inputRef = useRef<HTMLInputElement>(null);
  const feedRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const dockMouseX = useMotionValue(Infinity);

  // -- Derived --
  const isRunning = atlas.running || griffinPipelineRunning;
  const hasArtifacts = griffinArtifacts.length > 0;

  // Auto-open right panel + switch to terminal when pipeline starts, workstation when done
  useEffect(() => {
    if (griffinPipelineRunning) {
      setRightOpen(true);
      setRightTab("terminal");
    }
  }, [griffinPipelineRunning]);

  useEffect(() => {
    if (hasArtifacts) {
      setRightOpen(true);
      setRightTab("workstation");
    }
  }, [hasArtifacts]);

  // Auto-scroll feed
  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [atlas.progress, atlas.result, griffinChat, griffinAgentMsgs]);

  // -- Smart routing --
  const isComplexTask = useCallback((cmd: string) => {
    const turboKeywords = /\b(build|deploy|create app|create website|full.?stack|generate|scaffold|make me a|develop|launch|spin up|put together)\b/i;
    return turboKeywords.test(cmd) && griffinConnected;
  }, [griffinConnected]);

  // -- Voice --
  const toggleVoice = useCallback(() => {
    if (listening) { recognitionRef.current?.stop(); setListening(false); return; }
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRec) return;
    const rec = new SpeechRec();
    rec.continuous = false;
    rec.interimResults = true;
    rec.onresult = (e: any) => {
      const transcript = Array.from(e.results as any[]).map((r: any) => r[0].transcript).join("");
      setQuery(transcript);
      if (e.results[e.results.length - 1].isFinal) {
        setListening(false);
        if (transcript.trim()) submitCommand(transcript.trim());
      }
    };
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    rec.start();
    recognitionRef.current = rec;
    setListening(true);
  }, [listening]); // eslint-disable-line react-hooks/exhaustive-deps

  // -- Submit --
  const submitCommand = useCallback((cmd?: string) => {
    const text = (cmd ?? query).trim();
    if (!text) return;

    if (isComplexTask(text)) {
      griffinSend(text);
      setRightOpen(true);
    } else if (atlas.connected) {
      atlas.sendCommand(text);
    }
    if (!cmd) setQuery("");
  }, [query, isComplexTask, griffinSend, atlas]);

  // -- Keyboard --
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter") { e.preventDefault(); submitCommand(); }
    else if (e.key === "Escape") {
      if (atlas.running) atlas.stopAgent();
      else if (multiverse) setMultiverse(false);
      else if (showConn) setShowConn(false);
    }
  }, [submitCommand, atlas, multiverse, showConn]);

  // -- Status --
  const atlasStatus = atlas.connected;
  const wrapperCount = Object.keys(griffinWrappers).length;
  const activeWrappers = Object.values(griffinWrappers).filter(w => w.status === "WORKING" || w.status === "THINKING").length;

  // -- Dock items --
  const dockItems = useMemo(() => [
    { id: "blueprint" as RightTab, label: "Blueprint", icon: LayoutGrid },
    { id: "workstation" as RightTab, label: "Workstation", icon: Code2 },
    { id: "terminal" as RightTab, label: "Terminal", icon: Terminal },
    { id: "cost" as RightTab, label: "Costs", icon: DollarSign },
  ], []);

  return (
    <div
      className="relative w-full h-screen bg-[#0c0508] overflow-hidden flex flex-col"
      style={{
        backgroundImage: `
          radial-gradient(ellipse 90% 55% at 50% -5%, rgba(250,15,131,0.07) 0%, transparent 65%),
          radial-gradient(ellipse 45% 35% at 5% 95%, rgba(145,17,80,0.05) 0%, transparent 60%)
        `,
      }}
      onKeyDown={handleKeyDown}
    >
      {/* Dot-grid texture */}
      <div className="absolute inset-0 pointer-events-none" style={{
        backgroundImage: "radial-gradient(circle, rgba(230,147,188,0.08) 1px, transparent 1px)",
        backgroundSize: "32px 32px",
      }} />

      {/* ── Top Bar ────────────────────────────────────────────────────── */}
      <div className="relative z-20 flex items-center px-5 py-3 border-b border-[#911150]/15 shrink-0">
        {/* Logo */}
        <h1
          className="text-base font-black text-transparent bg-clip-text select-none"
          style={{
            backgroundImage: "linear-gradient(140deg, #f0e4ea 10%, #fa0f83 50%, #e693bc 90%)",
            fontFamily: "'Monocraft', 'Courier New', monospace",
            letterSpacing: "0.22em",
          }}
        >
          ATLAS
        </h1>

        {/* Status indicators */}
        <div className="ml-4 flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className={`w-1.5 h-1.5 rounded-full ${atlasStatus ? "bg-green-400 shadow-[0_0_6px_#4ade80]" : "bg-orange-400"}`} />
            <span className="text-[9px] font-mono text-[#e693bc]/40">
              agent {atlasStatus ? "live" : "off"}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className={`w-1.5 h-1.5 rounded-full ${griffinConnected ? "bg-[#fa0f83] shadow-[0_0_6px_#fa0f83]" : "bg-[#911150]/40"}`} />
            <span className="text-[9px] font-mono text-[#e693bc]/40">
              orchestrator {griffinConnected ? (activeWrappers > 0 ? `${activeWrappers} active` : griffinPipelineRunning ? "running" : "idle") : "off"}
            </span>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setRightOpen(v => !v)}
            className="p-1.5 rounded-lg text-[#e693bc]/30 hover:text-[#e693bc]/60 hover:bg-white/5 transition-all"
            title={rightOpen ? "Collapse panel" : "Expand panel"}
          >
            {rightOpen ? <PanelRightClose className="w-4 h-4" /> : <PanelRightOpen className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setShowConn(v => !v)}
            className="p-1.5 rounded-lg text-[#e693bc]/30 hover:text-[#e693bc]/60 hover:bg-white/5 transition-all"
            title="Connection settings"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ── Main Split ─────────────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-1 min-h-0 overflow-hidden">

        {/* ── Left Panel: Command + Feed ───────────────────────────────── */}
        <div className={`flex flex-col min-h-0 transition-all duration-300 ${rightOpen ? "w-[55%]" : "w-full"}`}>

          {/* Command input card */}
          <div className="px-5 pt-4 pb-2 shrink-0">
            <div
              className="rounded-2xl border border-[#911150]/30 bg-[#0f0306]/90 backdrop-blur-md overflow-hidden"
              style={{
                boxShadow: isRunning
                  ? "0 0 0 1px rgba(250,15,131,0.3), 0 0 40px rgba(250,15,131,0.08)"
                  : "0 0 0 1px rgba(145,17,80,0.15)",
              }}
            >
              <div className="flex items-center gap-2 px-4 py-3">
                <Zap className="w-4 h-4 text-[#fa0f83]/50 shrink-0" />
                <input
                  ref={inputRef}
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  autoFocus
                  disabled={isRunning}
                  placeholder={
                    isRunning
                      ? "Working..."
                      : !atlas.connected && !griffinConnected
                      ? "Waiting for backend..."
                      : "What do you want ATLAS to do?"
                  }
                  className="flex-1 bg-transparent text-[#f0e4ea] text-sm placeholder:text-[#e693bc]/20 outline-none disabled:opacity-40 font-mono"
                />
                {/* Mic */}
                <button
                  onClick={toggleVoice}
                  className={`p-1.5 rounded-lg transition-all ${
                    listening
                      ? "text-[#fa0f83] bg-[#fa0f83]/12 shadow-[0_0_12px_rgba(250,15,131,0.4)] animate-pulse"
                      : "text-[#e693bc]/26 hover:text-[#e693bc]/55 hover:bg-white/[0.04]"
                  }`}
                  title={listening ? "Stop" : "Voice"}
                >
                  <Mic className="w-3.5 h-3.5" />
                </button>
                {/* Send / Stop */}
                {isRunning ? (
                  <button
                    onClick={atlas.stopAgent}
                    className="px-2.5 py-1 rounded-lg text-[10px] font-mono bg-red-500/10 text-red-400/70 hover:bg-red-500/18 border border-red-500/22 transition-all"
                  >
                    stop
                  </button>
                ) : (
                  <button
                    onClick={() => submitCommand()}
                    disabled={!query.trim() || (!atlas.connected && !griffinConnected)}
                    className="p-1.5 rounded-lg bg-gradient-to-br from-[#fa0f83] to-[#911150] text-white disabled:opacity-20 disabled:cursor-not-allowed hover:shadow-[0_0_16px_rgba(250,15,131,0.45)] active:scale-95 transition-all"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                )}
              </div>
              {/* Route indicator */}
              {query.trim() && !isRunning && (
                <div className="px-4 pb-2 -mt-0.5">
                  <span className="text-[9px] font-mono text-[#e693bc]/25">
                    {isComplexTask(query)
                      ? "~ multi-agent orchestrator"
                      : "~ desktop agent"}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Activity Feed */}
          <div ref={feedRef} className="flex-1 overflow-y-auto px-5 pb-20 pt-2 space-y-2 min-h-0">

            {/* Empty state */}
            {!isRunning && atlas.progress.length === 0 && !atlas.result && griffinChat.length === 0 && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15 }}
                className="flex flex-col items-center py-16 gap-4"
              >
                <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-[#fa0f83]/15 to-[#911150]/10 border border-[#911150]/20 flex items-center justify-center">
                  <Zap className="w-5 h-5 text-[#fa0f83]/40" />
                </div>
                <div className="text-center space-y-2">
                  <p className="text-[#e693bc]/30 text-xs font-mono">
                    Tell ATLAS what to do
                  </p>
                  <div className="flex flex-wrap items-center justify-center gap-2">
                    {[
                      "Open Notepad and type hello",
                      "Build me a portfolio website",
                    ].map((ex) => (
                      <button
                        key={ex}
                        onClick={() => { setQuery(ex); inputRef.current?.focus(); }}
                        className="px-3 py-1.5 rounded-lg bg-[#911150]/10 border border-[#911150]/15 text-[#e693bc]/30 text-[10px] font-mono hover:bg-[#911150]/20 hover:text-[#e693bc]/50 transition-all"
                      >
                        &ldquo;{ex}&rdquo;
                      </button>
                    ))}
                  </div>
                  <p className="text-[#e693bc]/15 text-[9px] font-mono mt-3">
                    Simple tasks use the desktop agent - complex tasks use multi-agent orchestration
                  </p>
                </div>
              </motion.div>
            )}

            {/* Not connected banner */}
            {!atlas.connected && !griffinConnected && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-2.5 px-4 py-3 rounded-xl bg-[#fa0f83]/5 border border-[#fa0f83]/12 text-[#e693bc]/42 text-xs font-mono"
              >
                <Loader2 className="w-3.5 h-3.5 animate-spin text-[#fa0f83]/48 shrink-0" />
                Waiting for backend connections...
              </motion.div>
            )}

            {/* ATLAS agent progress cards */}
            <AnimatePresence>
              {atlas.progress.map((p, i) => (
                <motion.div
                  key={`atlas-${i}`}
                  initial={{ opacity: 0, y: 5, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: 0.18 }}
                  className="flex items-start gap-3 px-4 py-3 rounded-xl bg-[#0f0306]/80 border border-[#911150]/16"
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
                      <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-[#911150]/15 text-[#fa0f83]/50 uppercase tracking-wider">
                        agent
                      </span>
                      <span className="text-[9px] font-mono tracking-widest text-[#fa0f83]/52 uppercase">
                        {p.step}
                      </span>
                      {p.status && (
                        <span className={`text-[9px] font-mono uppercase tracking-wide px-1.5 py-0.5 rounded-full ${
                          p.status === "completed" ? "bg-green-500/10 text-green-400/65"
                            : p.status === "failed" ? "bg-red-500/10 text-red-400/65"
                            : "bg-[#fa0f83]/8 text-[#fa0f83]/58"
                        }`}>
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

            {/* ATLAS result */}
            {atlas.result && (
              <motion.div
                initial={{ opacity: 0, y: 8, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                className={`flex items-start gap-3 px-4 py-4 rounded-xl border ${
                  atlas.result.success
                    ? "bg-green-500/[0.06] border-green-500/20"
                    : "bg-red-500/[0.06] border-red-500/20"
                }`}
              >
                {atlas.result.success ? (
                  <CheckCircle2 className="w-4 h-4 text-green-400/75 shrink-0 mt-0.5" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-400/75 shrink-0 mt-0.5" />
                )}
                <div>
                  <p className={`text-xs font-mono font-medium ${
                    atlas.result.success ? "text-green-300/75" : "text-red-300/75"
                  }`}>
                    {atlas.result.success ? "Task completed successfully" : "Task failed"}
                  </p>
                  {atlas.result.detail && (
                    <p className={`text-xs font-mono mt-1 leading-relaxed ${
                      atlas.result.success ? "text-green-300/45" : "text-red-300/45"
                    }`}>
                      {atlas.result.detail}
                    </p>
                  )}
                </div>
              </motion.div>
            )}

            {/* Griffin chat messages (multi-agent) */}
            <AnimatePresence>
              {griffinChat.map((msg, i) => (
                <motion.div
                  key={`gchat-${i}`}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex gap-3 px-4 py-3 rounded-xl ${
                    msg.isUser
                      ? "bg-[#fa0f83]/[0.06] border border-[#fa0f83]/15"
                      : "bg-[#0f0306]/80 border border-[#911150]/16"
                  }`}
                >
                  <div className="mt-0.5 shrink-0">
                    {msg.isUser ? (
                      <div className="w-5 h-5 rounded-full bg-[#fa0f83]/20 flex items-center justify-center">
                        <ChevronRight className="w-3 h-3 text-[#fa0f83]/60" />
                      </div>
                    ) : (
                      <div className="w-5 h-5 rounded-full bg-[#911150]/30 flex items-center justify-center">
                        <Zap className="w-3 h-3 text-[#fa0f83]/60" />
                      </div>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-[#fa0f83]/10 text-[#fa0f83]/50 uppercase tracking-wider">
                        {msg.isUser ? "you" : "orchestrator"}
                      </span>
                    </div>
                    <p className="text-[#f0e4ea]/55 text-xs font-mono leading-relaxed break-words whitespace-pre-wrap">
                      {msg.content}
                    </p>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Griffin agent-to-agent messages */}
            <AnimatePresence>
              {griffinAgentMsgs.map((msg, i) => (
                <motion.div
                  key={`gagent-${i}`}
                  initial={{ opacity: 0, x: -5 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-start gap-2.5 px-3 py-2 rounded-lg bg-[#0f0306]/50 border border-[#911150]/10"
                >
                  <div className="w-4 h-4 rounded-full bg-[#911150]/20 flex items-center justify-center mt-0.5 shrink-0">
                    <Zap className="w-2.5 h-2.5 text-[#e693bc]/40" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <span className="text-[8px] font-mono text-[#e693bc]/35 uppercase tracking-wider">
                      {msg.author}
                    </span>
                    <p className="text-[#f0e4ea]/35 text-[11px] font-mono leading-relaxed break-words">
                      {msg.content}
                    </p>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* ── Right Panel: Context Tabs ────────────────────────────────── */}
        <AnimatePresence>
          {rightOpen && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: "45%", opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
              className="border-l border-[#911150]/15 flex flex-col min-h-0 overflow-hidden"
            >
              {/* Tab bar */}
              <div className="flex items-center gap-1 px-3 py-2 border-b border-[#911150]/12 shrink-0">
                {dockItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = rightTab === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setRightTab(item.id)}
                      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[10px] font-mono transition-all ${
                        isActive
                          ? "bg-[#fa0f83]/12 text-[#fa0f83] border border-[#fa0f83]/20"
                          : "text-[#e693bc]/35 hover:text-[#e693bc]/60 hover:bg-white/[0.03] border border-transparent"
                      }`}
                    >
                      <Icon className="w-3 h-3" />
                      {item.label}
                    </button>
                  );
                })}
              </div>

              {/* Tab content */}
              <div className="flex-1 min-h-0 overflow-hidden">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={rightTab}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    className="h-full w-full turbo-mode"
                  >
                    {rightTab === "blueprint" && <BlueprintCanvas />}
                    {rightTab === "workstation" && <Workstation />}
                    {rightTab === "terminal" && <GodModeTerminal />}
                    {rightTab === "cost" && <CostDashboard />}
                  </motion.div>
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Bottom Dock ────────────────────────────────────────────────── */}
      <motion.div
        onMouseMove={(e) => dockMouseX.set(e.pageX)}
        onMouseLeave={() => dockMouseX.set(Infinity)}
        className="fixed bottom-3 left-1/2 -translate-x-1/2 z-50 flex items-center gap-1 px-2 py-1.5 rounded-2xl border border-[#911150]/20 bg-[#0a0208]/80 backdrop-blur-xl shadow-2xl"
      >
        {dockItems.map((item) => (
          <DockIcon
            key={item.id}
            icon={item.icon}
            label={item.label}
            isActive={rightTab === item.id && rightOpen}
            onClick={() => {
              if (rightTab === item.id && rightOpen) {
                setRightOpen(false);
              } else {
                setRightTab(item.id);
                setRightOpen(true);
              }
            }}
            mouseX={dockMouseX}
          />
        ))}
        <div className="mx-0.5 h-5 w-px bg-[#911150]/20" />
        <DockIcon
          icon={Layers}
          label="Multiverse"
          isActive={multiverse}
          onClick={() => setMultiverse(true)}
          mouseX={dockMouseX}
        />
        <DockIcon
          icon={Settings}
          label="Settings"
          isActive={settingsOpen}
          onClick={() => setSettingsOpen(true)}
          mouseX={dockMouseX}
        />
      </motion.div>

      {/* ── Multiverse Overlay ─────────────────────────────────────────── */}
      <AnimatePresence>
        {multiverse && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[80] bg-[#0c0508]"
          >
            <MultiverseScene
              isOpen={true}
              onClose={() => setMultiverse(false)}
              onSelectUniverse={() => setMultiverse(false)}
            />
            <button
              onClick={() => setMultiverse(false)}
              className="fixed top-4 right-4 z-[90] px-3 py-1.5 rounded-lg bg-[#0a0208]/80 border border-[#911150]/30 text-[#e693bc]/50 text-xs font-mono hover:text-[#e693bc]/80 hover:bg-[#0a0208] transition-all backdrop-blur-md"
            >
              ESC to close
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Settings Dialog ────────────────────────────────────────────── */}
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />

      {/* ── Connection Panel ───────────────────────────────────────────── */}
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
              className="fixed right-0 top-0 bottom-0 w-80 z-50 flex flex-col shadow-2xl"
              style={{ background: "#0a0208", borderLeft: "1px solid rgba(250,15,131,0.15)" }}
            >
              <div className="flex items-start justify-between px-5 py-4 border-b border-[#911150]/20">
                <div>
                  <h2 className="text-[#f0e4ea]/88 text-sm font-semibold font-mono">Connections</h2>
                  <p className="text-[#e693bc]/32 text-[10px] mt-0.5 font-mono">Backend configuration</p>
                </div>
                <button onClick={() => setShowConn(false)} className="text-[#e693bc]/28 hover:text-[#e693bc]/65 transition-colors mt-0.5">
                  <XCircle className="w-4 h-4" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto px-5 py-5 space-y-6">
                {/* ATLAS Agent section */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${atlasStatus ? "bg-green-400 shadow-[0_0_6px_#4ade80]" : "bg-orange-400"}`} />
                    <span className="text-[#f0e4ea]/70 text-xs font-mono font-medium">Desktop Agent</span>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[#e693bc]/38 text-[9px] uppercase tracking-widest font-mono block">Host</label>
                    <input
                      type="text" value={wsHost} onChange={(e) => setWsHost(e.target.value)} placeholder="localhost"
                      className="w-full rounded-lg px-3 py-2.5 text-[#f0e4ea] text-xs font-mono outline-none placeholder:text-[#e693bc]/18"
                      style={{ background: "#0f0306", border: "1px solid rgba(145,17,80,0.28)" }}
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[#e693bc]/38 text-[9px] uppercase tracking-widest font-mono block">Port</label>
                    <input
                      type="number" value={wsPort} onChange={(e) => setWsPort(e.target.value)} placeholder="8000"
                      className="w-full rounded-lg px-3 py-2.5 text-[#f0e4ea] text-xs font-mono outline-none placeholder:text-[#e693bc]/18"
                      style={{ background: "#0f0306", border: "1px solid rgba(145,17,80,0.28)" }}
                    />
                  </div>
                  <div className="rounded-lg px-3 py-2 text-[10px] font-mono flex items-center gap-0.5" style={{ background: "rgba(250,15,131,0.04)", border: "1px solid rgba(145,17,80,0.2)" }}>
                    <span className="text-[#fa0f83]/45">ws://</span>
                    <span className="text-[#f0e4ea]/50">{wsHost || "localhost"}</span>
                    <span className="text-[#e693bc]/28">:</span>
                    <span className="text-[#f0e4ea]/50">{wsPort || "8000"}</span>
                    <span className="text-[#fa0f83]/45">/ws</span>
                  </div>
                </div>

                <div className="border-t border-[#911150]/15" />

                {/* Griffin Orchestrator section */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${griffinConnected ? "bg-[#fa0f83] shadow-[0_0_6px_#fa0f83]" : "bg-[#911150]/40"}`} />
                    <span className="text-[#f0e4ea]/70 text-xs font-mono font-medium">Multi-Agent Orchestrator</span>
                  </div>
                  <div className="rounded-lg px-3 py-2 text-[10px] font-mono" style={{ background: "rgba(250,15,131,0.04)", border: "1px solid rgba(145,17,80,0.2)" }}>
                    <span className="text-[#fa0f83]/45">ws://</span>
                    <span className="text-[#f0e4ea]/50">localhost:9100</span>
                  </div>
                  <p className="text-[#e693bc]/25 text-[9px] font-mono">
                    {griffinConnected
                      ? `Connected - ${wrapperCount} agents registered`
                      : "Not connected - start griffin ml-service"}
                  </p>
                </div>
              </div>

              <div className="px-5 pb-5 pt-3 border-t border-[#911150]/18">
                <button
                  onClick={() => setShowConn(false)}
                  className="w-full py-2.5 rounded-xl text-white text-xs font-medium font-mono active:scale-[0.98] transition-all"
                  style={{ background: "linear-gradient(135deg, #fa0f83, #911150)", boxShadow: "0 0 0 1px rgba(250,15,131,0.3)" }}
                >
                  Done
                </button>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
