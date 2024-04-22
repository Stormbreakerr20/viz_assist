import 'dart:math';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class SmartDeviceBox extends StatelessWidget {
  final String smartDeviceName;
  final String iconPath;
  final bool powerOn;
  void Function(bool)? onChanged;
  final int index;
  SmartDeviceBox(
      {super.key,
      required this.smartDeviceName,
      required this.iconPath,
      required this.powerOn,
      required this.onChanged,
      required this.index});

  void Handle(bool b) {
    print(b);
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(15.0),

          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(24),
              color: powerOn
                  ? Colors.grey[900]
                  : Color.fromARGB(44, 164, 167, 189),
            ),
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 25.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // icon
                  Image.asset(
                    iconPath,
                    height: (index == 0 || index == 3) ? 65 : 125,
                    color: powerOn ? Colors.white : Colors.grey.shade700,
                  ),

                  // smart device name + switch
                  Row(
                    children: [
                      Expanded(
                        child: Padding(
                          padding: const EdgeInsets.only(left: 25.0),
                          child: Text(
                            smartDeviceName,
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 20,
                              color: powerOn ? Colors.white : Colors.black,
                            ),
                          ),
                        ),
                      ),
                      Transform.rotate(
                        angle: pi / 2,
                        child: null,
                      ),
                    ],
                  )
                ],
              ),
            ),
          ),
          onTap: () {
            // Call the Handle function with your desired boolean value
            if (index == 0 || index == 3)
              Handle(true); // You can pass any boolean value here
          },
          splashColor: (index == 1 || index == 2)
              ? Colors.transparent
              : Color.fromRGBO(123, 125, 125, 100)),
    );
  }
}