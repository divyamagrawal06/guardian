import 'dart:async';
import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'services/atlas_service.dart';
import 'screens/connection_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AtlasApp());
}

class AtlasApp extends StatelessWidget {
  const AtlasApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: const AtlasHome(),
    );
  }
}

/// Helper widget to render tiny pixel-art PNGs scaled up with nearest-neighbor
/// filtering so they stay crisp instead of blurry.
class PixelArt extends StatelessWidget {
  const PixelArt({
    super.key,
    required this.asset,
    this.width,
    this.height,
    this.fit = BoxFit.contain,
  });

  final String asset;
  final double? width;
  final double? height;
  final BoxFit fit;

  @override
  Widget build(BuildContext context) {
    return Image.asset(
      asset,
      width: width,
      height: height,
      fit: fit,
      filterQuality: FilterQuality.none, // nearest-neighbor for pixel art
    );
  }
}

class AtlasHome extends StatefulWidget {
  const AtlasHome({super.key});

  @override
  State<AtlasHome> createState() => _AtlasHomeState();
}

class _AtlasHomeState extends State<AtlasHome> {
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  final AtlasService _service = AtlasService();
  final List<Map<String, dynamic>> _progress = [];
  Map<String, dynamic>? _result;

  StreamSubscription? _progressSub;
  StreamSubscription? _resultSub;
  StreamSubscription? _errorSub;

  // Speech-to-text (voice input like icarus-sense)
  final stt.SpeechToText _speech = stt.SpeechToText();
  bool _isListening = false;
  bool _speechAvailable = false;

  @override
  void initState() {
    super.initState();
    _initService();
  }

  Future<void> _initService() async {
    // Initialize speech recognition
    _speechAvailable = await _speech.initialize(
      onError: (error) {
        setState(() => _isListening = false);
      },
      onStatus: (status) {
        if (status == 'done' || status == 'notListening') {
          setState(() => _isListening = false);
        }
      },
    );

    await _service.loadSettings();

    // Listen to connection changes
    _service.addListener(_onServiceChanged);

    // Listen to progress events
    _progressSub = _service.progressStream.listen((event) {
      setState(() {
        _progress.add({
          'step': event.step,
          'status': event.status,
          'detail': event.detail,
        });
      });
      _scrollToBottom();
    });

    // Listen to results
    _resultSub = _service.resultStream.listen((result) {
      setState(() {
        _result = {
          'type': 'result',
          'success': result.success,
          'detail': result.detail,
        };
      });
      _scrollToBottom();
    });

    // Listen to errors
    _errorSub = _service.errorStream.listen((error) {
      setState(() {
        _result = {
          'type': 'error',
          'success': false,
          'detail': error,
        };
      });
      _scrollToBottom();
    });

    // Auto-connect if we have a saved IP
    if (_service.serverIp.isNotEmpty) {
      _service.connect();
    }
  }

  void _onServiceChanged() {
    if (mounted) setState(() {});
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _sendCommand() {
    final text = _controller.text.trim();
    if (text.isEmpty || !_service.isConnected || _service.agentRunning) return;

    setState(() {
      _progress.clear();
      _result = null;
    });

    _service.sendCommand(text);
  }

  void _stopAgent() {
    _service.stopAgent();
  }

  void _openSettings() async {
    await Navigator.of(context).push<bool>(
      MaterialPageRoute(
        builder: (_) => ConnectionScreen(service: _service),
      ),
    );
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _service.removeListener(_onServiceChanged);
    _progressSub?.cancel();
    _resultSub?.cancel();
    _errorSub?.cancel();
    _speech.stop();
    _service.dispose();
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // If no server configured, show connection screen
    if (_service.serverIp.isEmpty && !_service.isConnected) {
      return ConnectionScreen(service: _service);
    }

    return Scaffold(
      backgroundColor: Colors.white,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. Background Layer
          Positioned.fill(
            child: Image.asset(
              'assets/bg.png',
              fit: BoxFit.cover,
              filterQuality: FilterQuality.none,
            ),
          ),
          // 2. Content Layer
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 6.0),
              child: Column(
                children: [
                  const SizedBox(height: 5),
                  // Search Section with Owl
                  SizedBox(
                    height: 165,
                    child: Stack(
                      alignment: Alignment.topCenter,
                      clipBehavior: Clip.none,
                      children: [
                        Padding(
                          padding: const EdgeInsets.only(top: 25.0),
                          child: Stack(
                            clipBehavior: Clip.none,
                            children: [
                              Positioned.fill(
                                child: PixelArt(
                                  asset: 'assets/search_box.png',
                                  fit: BoxFit.contain,
                                ),
                              ),
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 25.0),
                                child: Row(
                                  children: [
                                    Expanded(
                                      child: TextField(
                                        controller: _controller,
                                        onSubmitted: (_) => _sendCommand(),
                                        decoration: const InputDecoration(
                                          hintText: "Ask ATLAS anything...",
                                          border: InputBorder.none,
                                          hintStyle: TextStyle(
                                            color: Color(0xFF4A4A4A),
                                            fontSize: 18,
                                            fontFamily: 'Courier',
                                          ),
                                        ),
                                        style: const TextStyle(
                                          fontSize: 18,
                                          fontFamily: 'Courier',
                                        ),
                                      ),
                                    ),
                                    // Mic button for voice input
                                    GestureDetector(
                                      onTap: _speechAvailable && !_service.agentRunning
                                          ? () {
                                              if (_isListening) {
                                                _speech.stop();
                                                setState(() => _isListening = false);
                                              } else {
                                                setState(() => _isListening = true);
                                                _speech.listen(
                                                  onResult: (result) {
                                                    setState(() {
                                                      _controller.text = result.recognizedWords;
                                                      _controller.selection = TextSelection.fromPosition(
                                                        TextPosition(offset: _controller.text.length),
                                                      );
                                                    });
                                                    if (result.finalResult && result.recognizedWords.isNotEmpty) {
                                                      setState(() => _isListening = false);
                                                      _sendCommand();
                                                    }
                                                  },
                                                  listenMode: stt.ListenMode.dictation,
                                                );
                                              }
                                            }
                                          : null,
                                      child: Icon(
                                        _isListening ? Icons.mic : Icons.mic_none,
                                        size: 22,
                                        color: _isListening
                                            ? const Color(0xFF4A5D3A)
                                            : const Color(0xFF4A4A4A),
                                      ),
                                    ),
                                    const SizedBox(width: 8),
                                    // Send / loading button
                                    GestureDetector(
                                      onTap: _service.agentRunning ? _stopAgent : _sendCommand,
                                      child: _service.agentRunning
                                          ? const SizedBox(
                                              width: 20,
                                              height: 20,
                                              child: CircularProgressIndicator(
                                                strokeWidth: 2,
                                                color: Color(0xFF4A4A4A),
                                              ),
                                            )
                                          : PixelArt(
                                              asset: 'assets/search_icon.png',
                                              height: 20,
                                              width: 20,
                                            ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                        // Owl on top
                        Positioned(
                          top: -12,
                          child: PixelArt(
                            asset: 'assets/owl.png',
                            height: 55,
                          ),
                        ),
                      ],
                    ),
                  ),

                  // Connection status + settings gear
                  Padding(
                    padding: const EdgeInsets.only(bottom: 4.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: _service.isConnected
                                ? Colors.green
                                : _service.status == ConnectionStatus.connecting
                                    ? Colors.orange
                                    : Colors.red.shade300,
                          ),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          _statusText(),
                          style: const TextStyle(
                            fontSize: 11,
                            color: Color(0xFF8F6B4F),
                            fontFamily: 'Courier',
                          ),
                        ),
                        const SizedBox(width: 8),
                        GestureDetector(
                          onTap: _openSettings,
                          child: Icon(
                            Icons.settings,
                            size: 16,
                            color: Colors.grey.shade500,
                          ),
                        ),
                      ],
                    ),
                  ),

                  // Progress / Result feed
                  Flexible(
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(maxHeight: 280),
                      child: _buildProgressFeed(),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _statusText() {
    switch (_service.status) {
      case ConnectionStatus.connected:
        return 'Connected to ${_service.serverIp}';
      case ConnectionStatus.connecting:
        return 'Connecting to ${_service.serverIp}...';
      case ConnectionStatus.disconnected:
        return 'Disconnected';
    }
  }

  Widget _buildProgressFeed() {
    if (!_service.agentRunning && _progress.isEmpty && _result == null) {
      return const SizedBox.shrink();
    }

    return Container(
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(12),
      ),
      child: ListView.builder(
        controller: _scrollController,
        padding: const EdgeInsets.all(12),
        itemCount: _progress.length + (_result != null ? 1 : 0),
        itemBuilder: (context, index) {
          // Final result
          if (index == _progress.length && _result != null) {
            final success = _result!['success'] == true;
            final message = _result!['detail'] as String? ??
                (success ? 'Task completed successfully!' : 'Task failed');
            
            return Container(
              padding: const EdgeInsets.all(12),
              margin: const EdgeInsets.only(top: 8, bottom: 4),
              decoration: BoxDecoration(
                color: success ? const Color(0xFF9CAF88).withValues(alpha: 0.5) : Colors.red.withValues(alpha: 0.3),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: success ? const Color(0xFF9CAF88) : Colors.red.shade400,
                  width: 1.5,
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    success ? Icons.check_circle : Icons.error,
                    color: success ? const Color(0xFF4A5D3A) : Colors.red.shade400,
                    size: 20,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: SingleChildScrollView(
                      scrollDirection: Axis.vertical,
                      child: Text(
                        message,
                        style: TextStyle(
                          fontSize: 14,
                          fontFamily: 'Courier',
                          color: success ? const Color(0xFF3A4A2E) : Colors.red.shade800,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            );
          }

          // Progress step
          final p = _progress[index];
          final step = p['step'] as String? ?? '';
          final status = p['status'] as String? ?? '';
          final detail = p['detail'] as String? ?? '';

          IconData icon;
          Color color;
          if (status == 'completed') {
            icon = Icons.check_circle;
            color = const Color(0xFF4A5D3A);
          } else if (status == 'failed' || status == 'retrying') {
            icon = Icons.error;
            color = const Color(0xFF4A5D3A);
          } else {
            icon = Icons.hourglass_top;
            color = const Color(0xFFA17F7A);
          }

          return Container(
            padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 8),
            margin: const EdgeInsets.only(bottom: 6),
            decoration: BoxDecoration(
              color: const Color(0xFFD1B79A).withValues(alpha: 0.6),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(icon, size: 16, color: color),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        step.toUpperCase(),
                        style: const TextStyle(
                          fontSize: 9,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFF8F6B4F),
                          letterSpacing: 1.2,
                        ),
                      ),
                      const SizedBox(height: 2),
                      ConstrainedBox(
                        constraints: const BoxConstraints(maxHeight: 75),
                        child: SingleChildScrollView(
                          child: Text(
                            detail,
                            style: const TextStyle(
                              fontSize: 13,
                              fontFamily: 'Courier',
                              height: 1.3,
                            ),
                            softWrap: true,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}