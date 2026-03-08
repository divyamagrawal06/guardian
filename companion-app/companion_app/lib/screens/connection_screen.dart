import 'package:flutter/material.dart';
import '../services/atlas_service.dart';

/// Screen shown when the app hasn't connected yet, or from the settings gear.
/// The user types their PC's local IP and taps "Connect".
class ConnectionScreen extends StatefulWidget {
  final AtlasService service;
  const ConnectionScreen({super.key, required this.service});

  @override
  State<ConnectionScreen> createState() => _ConnectionScreenState();
}

class _ConnectionScreenState extends State<ConnectionScreen> {
  final _ipController = TextEditingController();
  final _portController = TextEditingController();
  bool _testing = false;
  String? _testResult;

  @override
  void initState() {
    super.initState();
    _ipController.text = widget.service.serverIp;
    _portController.text = widget.service.serverPort.toString();
  }

  @override
  void dispose() {
    _ipController.dispose();
    _portController.dispose();
    super.dispose();
  }

  Future<void> _testAndConnect() async {
    final ip = _ipController.text.trim();
    final port = int.tryParse(_portController.text.trim()) ?? 8000;

    if (ip.isEmpty) {
      setState(() => _testResult = 'Please enter your PC\'s IP address');
      return;
    }

    setState(() {
      _testing = true;
      _testResult = null;
    });

    final reachable = await widget.service.testConnection(ip: ip, port: port);

    if (!mounted) return;

    if (reachable) {
      await widget.service.saveSettings(ip, port);
      widget.service.connect();
      setState(() {
        _testing = false;
        _testResult = '✓ Connected!';
      });
      // Pop back after short delay
      Future.delayed(const Duration(milliseconds: 600), () {
        if (mounted) Navigator.of(context).pop(true);
      });
    } else {
      setState(() {
        _testing = false;
        _testResult =
            '✗ Could not reach ATLAS at $ip:$port.\n'
            'Make sure the server is running and both devices are on the same Wi-Fi.';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F0E8),
      appBar: AppBar(
        title: const Text('Connect to ATLAS',
            style: TextStyle(fontFamily: 'Courier')),
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: const Color(0xFF4A4A4A),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Enter your PC\'s local IP address.\n'
              'Both your phone and PC must be on the same Wi-Fi network.',
              style: TextStyle(
                  fontSize: 14,
                  fontFamily: 'Courier',
                  color: Color(0xFF666666)),
            ),
            const SizedBox(height: 8),
            const Text(
              'Tip: Run "ipconfig" on your PC and look for IPv4 Address '
              '(usually 192.168.x.x)',
              style: TextStyle(
                  fontSize: 12,
                  fontFamily: 'Courier',
                  fontStyle: FontStyle.italic,
                  color: Color(0xFF888888)),
            ),
            const SizedBox(height: 24),

            // IP field
            TextField(
              controller: _ipController,
              keyboardType: TextInputType.numberWithOptions(decimal: true),
              decoration: InputDecoration(
                labelText: 'Server IP',
                hintText: '192.168.1.100',
                border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8)),
                prefixIcon: const Icon(Icons.computer),
              ),
              style: const TextStyle(fontFamily: 'Courier', fontSize: 18),
            ),
            const SizedBox(height: 12),

            // Port field
            TextField(
              controller: _portController,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: 'Port',
                hintText: '8000',
                border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8)),
                prefixIcon: const Icon(Icons.lan),
              ),
              style: const TextStyle(fontFamily: 'Courier', fontSize: 18),
            ),
            const SizedBox(height: 24),

            // Connect button
            SizedBox(
              width: double.infinity,
              height: 48,
              child: ElevatedButton.icon(
                onPressed: _testing ? null : _testAndConnect,
                icon: _testing
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.white))
                    : const Icon(Icons.link),
                label: Text(_testing ? 'Testing...' : 'Test & Connect'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF4A4A4A),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8)),
                  textStyle:
                      const TextStyle(fontFamily: 'Courier', fontSize: 16),
                ),
              ),
            ),

            if (_testResult != null) ...[
              const SizedBox(height: 16),
              Text(
                _testResult!,
                style: TextStyle(
                  fontFamily: 'Courier',
                  fontSize: 13,
                  color: _testResult!.startsWith('✓')
                      ? Colors.green.shade700
                      : Colors.red.shade700,
                ),
              ),
            ],

            const Spacer(),

            // Help section
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.6),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('How to find your PC\'s IP:',
                      style: TextStyle(
                          fontFamily: 'Courier',
                          fontWeight: FontWeight.bold,
                          fontSize: 13)),
                  SizedBox(height: 4),
                  Text(
                    '1. Open Command Prompt on your PC\n'
                    '2. Type: ipconfig\n'
                    '3. Look for "IPv4 Address" under Wi-Fi\n'
                    '4. Enter that IP above (e.g. 192.168.1.42)',
                    style: TextStyle(fontFamily: 'Courier', fontSize: 12),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
