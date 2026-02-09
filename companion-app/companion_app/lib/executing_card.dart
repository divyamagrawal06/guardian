import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class ExecutingCard extends StatelessWidget {
  final String message;

  const ExecutingCard({
    super.key,
    required this.message,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Your pixel-art / svg / image
          
          SvgPicture.asset(
            'assets/owl.svg',
            width: 80,
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: const Color(0xFFFFF2CC),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              message,
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontFamily: 'monospace',
              ),
            ),
          ),
          const SizedBox(height: 8),
          const Text('Executing...'),
        ],
      ),
    );
  }
}
