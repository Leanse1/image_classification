import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteHelper {
  static Interpreter? _interpreter;
  static bool _isInitialized = false;

  /// Initialize Interpreter
  static Future<void> init() async {
    try {
      final modelData =
          await rootBundle.load('assets/efficientnet_best_float32.tflite');

      if (modelData.lengthInBytes == 0) {
        throw Exception('Model file is empty!');
      }

      _interpreter = Interpreter.fromBuffer(modelData.buffer.asUint8List());
      _isInitialized = true;
      print("Model loaded successfully!");
    } catch (e) {
      _isInitialized = false;
      print("Failed to load model: $e");
    }
  }

  static Interpreter get interpreter {
    if (!_isInitialized || _interpreter == null) {
      throw Exception("Model not initialized. Call TFLiteHelper.init() first.");
    }
    return _interpreter!;
  }

  /// Normalize + resize image
  static List<List<List<List<double>>>> preprocessImage(File imageFile) {
    final image = img.decodeImage(imageFile.readAsBytesSync());
    if (image == null) throw Exception("Failed to decode image!");

    final resized = img.copyResize(image, width: 224, height: 224);

    final input = List.generate(
      1,
      (_) => List.generate(
        224,
        (_) => List.generate(
          224,
          (_) => List.filled(3, 0.0),
        ),
      ),
    );

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final p = resized.getPixel(x, y);

        input[0][y][x][0] = (p.r - 127.5) / 127.5;
        input[0][y][x][1] = (p.g - 127.5) / 127.5;
        input[0][y][x][2] = (p.b - 127.5) / 127.5;
      }
    }

    return input;
  }

  /// Classification
  static Future<String> classifyImage(File imageFile) async {
    if (!_isInitialized || _interpreter == null) {
      throw Exception("Model not initialized.");
    }

    final input = preprocessImage(imageFile);
    final output = List.generate(1, (_) => List.filled(7, 0.0));

    try {
      _interpreter!.run(input, output);
    } catch (e) {
      throw Exception("Inference failed: $e");
    }

    print(output); // logits

    /// Convert logits → probabilities
    final probs = softmax(output[0]);
    print(probs);

    /// Find max probability
    double maxScore = probs[0];
    int maxIndex = 0;

    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > maxScore) {
        maxScore = probs[i];
        maxIndex = i;
      }
    }

    /// Load labels
    final labels =
        (await rootBundle.loadString('assets/labels_efficientnet.txt'))
            .split('\n');

    final label = labels[maxIndex].trim();
    final confidence = (maxScore * 100).toStringAsFixed(2);

    return "$label - $confidence%";
  }
}

/// Softmax
List<double> softmax(List<double> logits) {
  final maxVal = logits.reduce(math.max);
  final exps = logits.map((e) => math.exp(e - maxVal)).toList();
  final sum = exps.reduce((a, b) => a + b);
  return exps.map((e) => e / sum).toList();
}
