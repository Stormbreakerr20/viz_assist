import 'dart:math';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class SmartDeviceBox extends StatelessWidget {
  final String smartDeviceName;
  final String iconPath;
  final bool powerOn;
  final void Function(bool)? onChanged;
  final int index;
  final Color? customColor; // New attribute for custom color

  const SmartDeviceBox({
    super.key, // Added key parameter
    required this.smartDeviceName,
    required this.iconPath,
    required this.powerOn,
    required this.onChanged,
    required this.index,
    this.customColor, // Initialized customColor parameter
  }); // Calling super constructor with key parameter

  void handle(bool b) {
    if (onChanged != null) {
      onChanged!(b);
    }
  }

  @override
  Widget build(BuildContext context) {
    final double verticalPadding = smartDeviceName.isEmpty ? 30.0 : 40.0;

    return Padding(
      padding: const EdgeInsets.all(15.0),
      child: GestureDetector(
        onTap: () {
          if (index == 0 || index == 3) {
            handle(true);
          }
        },
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(24),
            gradient: customColor == null // Checking if customColor is provided
                ? LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [
                powerOn ? const Color(0xFF4D90A6) : const Color(0xFF539FB8),
                powerOn ? const Color(0xFF01516C) : const Color(0xFF006789),
              ],
            )
                : null, // Setting gradient to null if customColor is provided
            color: customColor, // Using customColor if provided
          ),
          child: Padding(
            padding: EdgeInsets.symmetric(vertical: verticalPadding),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Padding(
                  padding: const EdgeInsets.only(bottom: 8.0),
                  child: Image.asset(
                    iconPath,
                    height: (index == 0 || index == 3) ? 65 : 125,
                    color: powerOn ? Colors.white : Colors.white60,
                  ),
                ),
                Row(
                  children: [
                    Expanded(
                      child: Padding(
                        padding: const EdgeInsets.only(left: 25.0),
                        child: Text(
                          smartDeviceName,
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 24,
                            color: powerOn ? Colors.white : Colors.white60,
                          ),
                        ),
                      ),
                    ),
                  ],
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
  }