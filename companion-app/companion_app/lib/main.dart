import 'package:flutter/material.dart';

void main() {
  runApp(const AtlasApp());
}

class AtlasApp extends StatelessWidget {
  const AtlasApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: AtlasHome(),
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

class AtlasHome extends StatelessWidget {
  const AtlasHome({super.key});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 2. Content Layer
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 10.0),
              child: Column(
                children: [
                  const SizedBox(height: 10),
                  // Search Section
                  SizedBox(
                    height: 200, // Constrain search section height
                    child: Stack(
                      alignment: Alignment.topCenter,
                      clipBehavior: Clip.none, // Allows owl to overflow box
                      children: [
                        // The Search Box Wrapper
                        Padding(
                          padding: const EdgeInsets.only(top: 25.0), // Space for owl
                          child: Stack(
                            clipBehavior: Clip.none,
                            children: [
                              // The Search Box Image
                              Positioned.fill(
                                child: PixelArt(
                                  asset: 'assets/search_box.png',
                                  fit: BoxFit.contain,
                                ),
                              ),
                              // Text and Icon inside the box
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 25.0),
                                child: Row(
                                  children: [
                                    const Expanded(
                                      child: TextField(
                                        decoration: InputDecoration(
                                          hintText: "Ask ATLAS anything...",
                                          border: InputBorder.none,
                                          hintStyle: TextStyle(
                                            color: Color(0xFF4A4A4A),
                                            fontSize: 18,
                                            fontFamily: 'Courier',
                                          ),
                                        ),
                                      ),
                                    ),
                                    PixelArt(
                                      asset: 'assets/search_icon.png',
                                      height: 20,
                                      width: 20,
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                        // The Owl sitting on top
                        Positioned(
                          top: 0,
                          child: PixelArt(
                            asset: 'assets/owl.png',
                            height: 55,
                          ),
                        ),
                      ],
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
}