import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

/// Persistent keys
const _kServerIp = 'atlas_server_ip';
const _kServerPort = 'atlas_server_port';

/// Represents the current connection state.
enum ConnectionStatus { disconnected, connecting, connected }

/// A single progress event from the backend.
class AgentProgressEvent {
  final String step;
  final String status;
  final String detail;
  AgentProgressEvent(
      {required this.step, required this.status, this.detail = ''});

  factory AgentProgressEvent.fromJson(Map<String, dynamic> json) =>
      AgentProgressEvent(
        step: json['step'] as String? ?? '',
        status: json['status'] as String? ?? '',
        detail: json['detail'] as String? ?? '',
      );
}

/// A final result from the backend.
class AgentResult {
  final bool success;
  final String detail;
  AgentResult({required this.success, this.detail = ''});
}

/// Central service that manages the WebSocket connection from the companion
/// app to the ATLAS backend running on the user's PC.
class AtlasService extends ChangeNotifier {
  // ── Stored settings ────────────────────────────────────────────────────
  String _serverIp = '';
  int _serverPort = 8000;

  String get serverIp => _serverIp;
  int get serverPort => _serverPort;

  String get wsUrl => 'ws://$_serverIp:$_serverPort/ws';
  String get httpBase => 'http://$_serverIp:$_serverPort';

  // ── Connection state ───────────────────────────────────────────────────
  ConnectionStatus _status = ConnectionStatus.disconnected;
  ConnectionStatus get status => _status;

  bool get isConnected => _status == ConnectionStatus.connected;

  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  Timer? _reconnectTimer;
  Timer? _pingTimer;

  // ── Streams exposed to the UI ──────────────────────────────────────────
  final _progressController =
      StreamController<AgentProgressEvent>.broadcast();
  Stream<AgentProgressEvent> get progressStream => _progressController.stream;

  final _resultController = StreamController<AgentResult>.broadcast();
  Stream<AgentResult> get resultStream => _resultController.stream;

  final _errorController = StreamController<String>.broadcast();
  Stream<String> get errorStream => _errorController.stream;

  bool _agentRunning = false;
  bool get agentRunning => _agentRunning;

  bool _agentReady = false;
  bool get agentReady => _agentReady;

  // ── Init from SharedPreferences ────────────────────────────────────────

  Future<void> loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    _serverIp = prefs.getString(_kServerIp) ?? '127.0.0.1';
    _serverPort = prefs.getInt(_kServerPort) ?? 8000;
    notifyListeners();
  }

  Future<void> saveSettings(String ip, int port) async {
    _serverIp = ip.trim();
    _serverPort = port;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kServerIp, _serverIp);
    await prefs.setInt(_kServerPort, _serverPort);
    notifyListeners();
  }

  // ── Connection management ──────────────────────────────────────────────

  /// Test whether the backend is reachable (HTTP health check).
  Future<bool> testConnection({String? ip, int? port}) async {
    final testIp = ip ?? _serverIp;
    final testPort = port ?? _serverPort;
    if (testIp.isEmpty) return false;
    try {
      final uri = Uri.parse('http://$testIp:$testPort/health');
      final resp = await http.get(uri).timeout(const Duration(seconds: 3));
      if (resp.statusCode == 200) {
        final body = jsonDecode(resp.body) as Map<String, dynamic>;
        return body['status'] == 'ok';
      }
    } catch (_) {}
    return false;
  }

  /// Open the WebSocket and start listening.
  void connect() {
    if (_serverIp.isEmpty) return;
    disconnect(); // clean previous

    _status = ConnectionStatus.connecting;
    notifyListeners();

    try {
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));

      _subscription = _channel!.stream.listen(
        _onMessage,
        onDone: _onDisconnected,
        onError: (_) => _onDisconnected(),
      );

      // Ping every 15s to keep the connection alive
      _pingTimer?.cancel();
      _pingTimer = Timer.periodic(const Duration(seconds: 15), (_) {
        if (_status == ConnectionStatus.connected) {
          try {
            _channel?.sink.add(jsonEncode({'type': 'ping'}));
          } catch (_) {}
        }
      });
    } catch (_) {
      _onDisconnected();
    }
  }

  void disconnect() {
    _reconnectTimer?.cancel();
    _pingTimer?.cancel();
    _subscription?.cancel();
    _channel?.sink.close();
    _channel = null;
    _status = ConnectionStatus.disconnected;
    _agentReady = false;
    _agentRunning = false;
    notifyListeners();
  }

  // ── Commands ───────────────────────────────────────────────────────────

  void sendCommand(String command) {
    if (!isConnected || command.trim().isEmpty) return;
    _agentRunning = true;
    notifyListeners();
    _channel?.sink.add(jsonEncode({'type': 'command', 'command': command}));
  }

  void stopAgent() {
    _channel?.sink.add(jsonEncode({'type': 'stop'}));
    _agentRunning = false;
    notifyListeners();
  }

  // ── Message handling ───────────────────────────────────────────────────

  void _onMessage(dynamic raw) {
    final msg = jsonDecode(raw as String) as Map<String, dynamic>;
    final type = msg['type'] as String? ?? '';

    switch (type) {
      case 'connected':
        _status = ConnectionStatus.connected;
        _agentReady = (msg['agent_ready'] as bool?) ?? false;
        notifyListeners();
        break;
      case 'progress':
        _progressController.add(AgentProgressEvent.fromJson(msg));
        // If we see a task-level status, update running flag
        if (msg['step'] == 'task') {
          final s = msg['status'] as String? ?? '';
          if (s == 'completed' || s == 'failed') {
            _agentRunning = false;
            notifyListeners();
          }
        }
        break;
      case 'result':
        final success = msg['success'] == true;
        _resultController.add(AgentResult(
          success: success,
          detail: msg['detail'] as String? ?? '',
        ));
        _agentRunning = false;
        notifyListeners();
        break;
      case 'error':
        _errorController.add(msg['message'] as String? ?? 'Unknown error');
        _agentRunning = false;
        notifyListeners();
        break;
      case 'stopped':
        _agentRunning = false;
        notifyListeners();
        break;
      case 'pong':
        break; // keepalive ack
    }
  }

  void _onDisconnected() {
    _status = ConnectionStatus.disconnected;
    _agentReady = false;
    _agentRunning = false;
    notifyListeners();

    // Auto-reconnect after 3s
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 3), () {
      if (_serverIp.isNotEmpty) connect();
    });
  }

  @override
  void dispose() {
    disconnect();
    _progressController.close();
    _resultController.close();
    _errorController.close();
    super.dispose();
  }
}
